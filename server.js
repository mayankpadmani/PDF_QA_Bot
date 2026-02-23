const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const path = require("path");
const rateLimit = require("express-rate-limit");
const session = require("express-session");

const app = express();
app.set("trust proxy", 1);
app.use(cors());
app.set('trust proxy', 1); // Fix ERR_ERL_UNEXPECTED_X_FORWARDED_FOR
app.use(express.json());

// Session middleware for per-user chat history
app.use(
  session({
    secret: "pdf-qa-bot-secret-key",
    resave: false,
    saveUninitialized: true,
    cookie: {
      secure: false,
      maxAge: 1000 * 60 * 60 * 24, // 24 hours
    },
  }),
);

// Rate limiting middleware
const uploadLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 5,
  message:
    "Too many PDF uploads from this IP, please try again after 15 minutes",
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
});

const askLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 30,
  message: "Too many questions asked, please try again after 15 minutes",
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
});

const summarizeLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
  message:
    "Too many summarization requests, please try again after 15 minutes",
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
});

const compareLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
  message:
    "Too many comparison requests, please try again after 15 minutes",
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
});

// Storage for uploaded PDFs
const upload = multer({ dest: "uploads/" });

app.post("/upload", uploadLimiter, upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res
        .status(400)
        .json({ error: "No file uploaded. Use form field name 'file'." });
    }

    const sessionId = req.body.sessionId;
    if (!sessionId) {
      return res.status(400).json({ error: "Missing sessionId." });
    }

    const filePath = path.join(__dirname, req.file.path);

    // Send PDF to Python service with session isolation
    const response = await axios.post("http://localhost:5000/process-pdf", {
      filePath: filePath,
      session_id: sessionId,
    });

    res.json({ doc_id: response.data.doc_id });
  } catch (err) {
    res.status(500).json({ error: "Upload failed" });
  }
});

app.post("/ask", askLimiter, async (req, res) => {
  const { question, sessionId } = req.body;
  if (!sessionId) {
    return res.status(400).json({ error: "Missing sessionId." });
  }

  try {
    // Initialize session chat history if it doesn't exist
    if (!req.session.chatHistory) {
      req.session.chatHistory = [];
    }

    // Add user message to session history
    req.session.chatHistory.push({
      role: "user",
      content: question,
    });

    // Send question + history to FastAPI with session isolation
    const response = await axios.post("http://localhost:5000/ask", {
      question: question,
      session_id: sessionId,
      history: req.session.chatHistory,
    });

    // Add assistant response to session history
    req.session.chatHistory.push({
      role: "assistant",
      content: response.data.answer,
    });

    res.json(response.data);
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: "Error asking question" });
  }
  res.json({ message: "History cleared" });
});

app.post("/clear-history", (req, res) => {
  // Clear only this user's session history
  if (req.session) {
    req.session.chatHistory = [];
  }
  res.json({ message: "History cleared" });
});

app.post("/summarize", summarizeLimiter, async (req, res) => {
  const { pdf, sessionId } = req.body || {};
  if (!sessionId) {
    return res.status(400).json({ error: "Missing sessionId." });
  }

  try {
    const response = await axios.post("http://localhost:5000/summarize", {
      pdf,
      session_id: sessionId,
    });
    res.json({ summary: response.data.summary });
  } catch (err) {
    console.error(err.response?.data || err.message);
    res.status(500).json({ error: "Error summarizing PDF" });
  }
});

app.post("/compare", compareLimiter, async (req, res) => {
  try {
    const response = await axios.post(
      "http://localhost:5000/compare",
      req.body,
    );
    res.json({ comparison: response.data.comparison });
  } catch (err) {
    console.error(err.response?.data || err.message);
    res.status(500).json({ error: "Error comparing documents" });
  }
});

app.listen(4000, () => console.log("Backend running on http://localhost:4000"));