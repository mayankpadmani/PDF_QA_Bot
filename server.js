const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const path = require("path");
const {
  globalLimiter,
  uploadLimiter,
  askLimiter,
  summarizeLimiter,
  compareLimiter,
} = require('./middleware/rateLimiter');

const app = express();
app.set('trust proxy', 1);
app.use(globalLimiter);
app.use(cors());
app.use(express.json());

const upload = multer({ dest: "uploads/" });

app.post("/upload", uploadLimiter, upload.single("file"), async (req, res) => {
  try {
    const filePath = path.join(__dirname, req.file.path);
    const response = await axios.post("http://localhost:5000/process-pdf", {
      filePath,
    });

    res.json({ doc_id: response.data.doc_id });
  } catch (err) {
    res.status(500).json({ error: "Upload failed" });
  }
});

app.post("/ask", askLimiter, async (req, res) => {
  const response = await axios.post("http://localhost:5000/ask", req.body);
  res.json(response.data);
});

app.post("/summarize", summarizeLimiter, async (req, res) => {
  const response = await axios.post("http://localhost:5000/summarize", req.body);
  res.json(response.data);
});

app.post("/compare", compareLimiter, async (req, res) => {
  try {
    const response = await axios.post("http://localhost:5000/compare", req.body);
    res.json({ comparison: response.data.comparison });
  } catch (err) {
    console.error(err.response?.data || err.message);
    res.status(500).json({ error: "Error comparing documents" });
  }
});

app.listen(4000, () => console.log("Backend running on http://localhost:4000"));