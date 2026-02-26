const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const axiosRetry = require("axios-retry").default;
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const rateLimit = require("express-rate-limit");
const { fileTypeFromFile } = require("file-type");

const app = express(); // Trust first proxy for rate limiting if behind a proxy
const session = require("express-session");
require("dotenv").config();

const app = express();

// ------------------------------------------------------------------
// CONFIGURATION
// ------------------------------------------------------------------
const API_REQUEST_TIMEOUT = parseInt(
  process.env.API_REQUEST_TIMEOUT || "45000",
  10
);

const MAX_RETRY_ATTEMPTS = parseInt(
  process.env.MAX_RETRY_ATTEMPTS || "3",
  10
);

// ------------------------------------------------------------------
// APP SETUP
// ------------------------------------------------------------------
app.set("trust proxy", 1);
app.use(cors());
app.use(express.json());

// ------------------------------------------------------------------
// SESSION (per-user chat history)
// ------------------------------------------------------------------
app.use(
  session({
    secret: "pdf-qa-bot-secret-key",
    resave: false,
    saveUninitialized: true,
    cookie: {
      secure: false,
      maxAge: 1000 * 60 * 60 * 24,
    },
  })
);

// ------------------------------------------------------------------
// AXIOS RETRY CONFIG
// ------------------------------------------------------------------
axiosRetry(axios, {
  retries: MAX_RETRY_ATTEMPTS,
  retryDelay: axiosRetry.exponentialDelay,
  retryCondition: (error) =>
    axiosRetry.isNetworkOrIdempotentRequestError(error) ||
    error.code === "ECONNABORTED" ||
    (error.response && error.response.status >= 500),
});

// ------------------------------------------------------------------
// RATE LIMITERS
// ------------------------------------------------------------------
const uploadLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 5,
});

const askLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 30,
});

const summarizeLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
});

const compareLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
});

// ------------------------------------------------------------------
// FILE STORAGE
// ------------------------------------------------------------------
const UPLOAD_DIR = path.resolve(__dirname, "uploads");

if (!fs.existsSync(UPLOAD_DIR)) {
  fs.mkdirSync(UPLOAD_DIR);
}

// ------------------------------------------------------------------
// MULTER CONFIG
// ------------------------------------------------------------------

const SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"];
const PDF_MIME_TYPE = "application/pdf";
const PDF_MAGIC = "%PDF"

const storage = multer.diskStorage({
  destination: "uploads/",
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const uniqueName = `${Date.now()}-${Math.round(
      Math.random() * 1e9
    )}${ext}`;
    cb(null, uniqueName);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 20 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const safeName = path.basename(file.originalname);
    const ext = path.extname(safeName).toLowerCase();
    if (SUPPORTED_EXTENSIONS.includes(ext) && file.mimetype === PDF_MIME_TYPE) {
      cb(null, true);
    } else {
      cb(new Error("Only PDF files are supported."));
    }
  },
});


const uploadSingle = (req, res, next) => {
  upload.single("file")(req, res, (err) => {
    if (!err) {
      return next();
    }

    if (err instanceof multer.MulterError) {
      return res.status(400).json({ error: err.message });
    }

    return res.status(400).json({ error: err.message || "Invalid upload." });
  });
};

const hasPdfMagicNumber = async (filePath) => {
  const handle = await fs.promises.open(filePath, "r");
  try {
    const buffer = Buffer.alloc(4);
    const { bytesRead } = await handle.read(buffer, 0, 4, 0);
    return bytesRead === 4 && buffer.toString("ascii") === PDF_MAGIC;
  } finally {
    await handle.close();
  }
}

// ------------------------------------------------------------------
// ROUTE: UPLOAD
// ------------------------------------------------------------------
app.post("/upload", uploadLimiter, uploadSingle, async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: "No file uploaded. Use form field name 'file'.",
      });
    }

    const { sessionId } = req.body;
    if (!sessionId) {
      return res.status(400).json({ error: "Missing sessionId." });
    }

    // **CRITICAL**: Clear session and reset backend state before processing new PDF
    // This prevents cross-document context leakage
    if (req.session) {
      req.session.chatHistory = [];
      req.session.currentPdfSessionId = null;
    }

    // Reset backend state through the /reset endpoint
    try {
      await axios.post("http://localhost:5000/reset");
    } catch (resetError) {
      console.warn("Warning: Could not reset backend state:", resetError.message);
      // Continue with PDF upload even if reset fails
    }

    // Send PDF to Python service for processing
    const uploadResponse = await axios.post("http://localhost:5000/process-pdf", {
      filePath: filePath,
    });

    // Store the new PDF session ID for future validation
    if (uploadResponse.data.session_id && req.session) {
      req.session.currentPdfSessionId = uploadResponse.data.session_id;
    }

    res.json({
      message: "PDF uploaded & processed successfully!",
      session_id: uploadResponse.data.session_id,
      details: uploadResponse.data
    });
  } catch (err) {
    console.error("Upload failed:", err.message);
    res.status(500).json({ error: "Upload failed" });
  }
});

// ------------------------------------------------------------------
// ROUTE: ASK
// ------------------------------------------------------------------
app.post("/ask", askLimiter, async (req, res) => {
  const { question, sessionId } = req.body;

  if (!sessionId)
    return res.status(400).json({ error: "Missing sessionId." });

  if (!question || typeof question !== "string" || !question.trim())
    return res.status(400).json({ error: "Invalid question." });

  try {
    if (!req.session.chatHistory) {
      req.session.chatHistory = [];
    }

    req.session.chatHistory.push({
      role: "user",
      content: question.trim(),
    });

    const response = await axios.post(
      "http://localhost:5000/ask",
      {
        question: question.trim(),
        session_id: sessionId,
        history: req.session.chatHistory,
      },
      { timeout: API_REQUEST_TIMEOUT }
    );

    req.session.chatHistory.push({
      role: "assistant",
      content: response.data.answer,
    });

    res.json(response.data);
  } catch (error) {
    console.error("Ask failed:", error.message);
    res.status(500).json({ error: "Error asking question" });
  }
});

// ------------------------------------------------------------------
// ROUTE: CLEAR HISTORY
// ------------------------------------------------------------------
app.post("/clear-history", (req, res) => {
  if (req.session) {
    req.session.chatHistory = [];
    req.session.currentPdfSessionId = null;
  }
  res.json({ message: "Chat history cleared" });
});

app.get("/pdf-status", async (req, res) => {
  try {
    // Check backend PDF status
    const statusResponse = await axios.get("http://localhost:5000/status");
    
    // Include frontend session status
    const frontendStatus = {
      hasSession: !!req.session,
      hasHistory: req.session?.chatHistory?.length > 0 || false,
      historyLength: req.session?.chatHistory?.length || 0,
      currentSessionId: req.session?.currentPdfSessionId || null
    };

    res.json({
      backend: statusResponse.data,
      frontend: frontendStatus
    });
  } catch (err) {
    console.error("Error fetching PDF status:", err.message);
    res.status(500).json({ error: "Could not fetch PDF status" });
  }
});

// ------------------------------------------------------------------
// ROUTE: SUMMARIZE
// ------------------------------------------------------------------
app.post("/summarize", summarizeLimiter, async (req, res) => {
  const { sessionId } = req.body;

  if (!sessionId)
    return res.status(400).json({ error: "Missing sessionId." });

  try {
    const response = await axios.post(
      "http://localhost:5000/summarize",
      { session_id: sessionId },
      { timeout: API_REQUEST_TIMEOUT }
    );

    res.json({ summary: response.data.summary });
  } catch (err) {
    if (err.code === "ECONNABORTED") {
      return res.status(504).json({ error: "Summarization timed out" });
    }
    res.status(500).json({ error: "Error summarizing" });
  }
});

// ------------------------------------------------------------------
// ROUTE: COMPARE
// ------------------------------------------------------------------
app.post("/compare", compareLimiter, async (req, res) => {
  try {
    const response = await axios.post(
      "http://localhost:5000/compare",
      req.body,
      { timeout: API_REQUEST_TIMEOUT }
    );

    res.json({ comparison: response.data.comparison });
  } catch (err) {
    res.status(500).json({ error: "Error comparing" });
  }
});

// ------------------------------------------------------------------
// ERROR HANDLING
// ------------------------------------------------------------------
app.use((err, req, res, next) => {
  if (err.code === "LIMIT_FILE_SIZE") {
    return res.status(400).json({
      error: "File too large (max 20MB).",
    });
  }
  if (err.message.includes("Unsupported file type")) {
    return res.status(400).json({ error: err.message });
  }
  next(err);
});

// ------------------------------------------------------------------
app.listen(4000, () =>
  console.log("Backend running on http://localhost:4000")
);