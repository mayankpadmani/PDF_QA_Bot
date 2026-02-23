
import React, { useState, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import { Document, Page, pdfjs } from "react-pdf";
import "bootstrap/dist/css/bootstrap.min.css";
import {
  Container,
  Row,
  Col,
  Button,
  Form,
  Card,
  Spinner,
  Navbar,
} from "react-bootstrap";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

const API_BASE = process.env.REACT_APP_API_URL || "";

// Theme persistence key
const THEME_STORAGE_KEY = "pdf-qa-bot-theme";

function App() {
  const [file, setFile] = useState(null);
  const [pdfs, setPdfs] = useState([]); // {name, doc_id, url}
  const [selectedDocs, setSelectedDocs] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [processingPdf, setProcessingPdf] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
    return savedTheme ? JSON.parse(savedTheme) : false;
  });
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [summarizing, setSummarizing] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [sessionId, setSessionId] = useState("");

  // Generate a session ID on mount (fix-data-leakage)
  useEffect(() => {
    setSessionId(
      crypto.randomUUID
        ? crypto.randomUUID()
        : Math.random().toString(36).substring(2, 15),
    );
  }, []);

  // Save theme preference when it changes
  useEffect(() => {
    localStorage.setItem(THEME_STORAGE_KEY, JSON.stringify(darkMode));
    document.body.classList.toggle("dark-mode", darkMode);
  }, [darkMode]);

  // ===============================
  // Upload
  // ===============================
  const uploadPDF = async () => {
    if (!file) return;
    setUploading(true);
    setProcessingPdf(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("sessionId", sessionId);
    try {
      const res = await axios.post(`${API_BASE}/upload`, formData);
      const url = URL.createObjectURL(file);
      setPdfs((prev) => [
        ...prev,
        { name: file.name, doc_id: res.data.doc_id, url },
      ]);
      setFile(null);
      alert("PDF uploaded!");
    } catch {
      alert("Upload failed.");
    }
    setUploading(false);
    setProcessingPdf(false);
  };

  // ===============================
  // Toggle selection
  // ===============================
  const toggleDocSelection = (doc_id) => {
    setComparisonResult(null);
    setSelectedDocs((prev) =>
      prev.includes(doc_id)
        ? prev.filter((id) => id !== doc_id)
        : [...prev, doc_id],
    );
  };

  // ===============================
  // Ask
  // ===============================
  const askQuestion = async () => {
    if (!question.trim() || selectedDocs.length === 0) return;
    setChatHistory((prev) => [...prev, { role: "user", text: question }]);
    setQuestion("");
    setAsking(true);
    try {
      const res = await axios.post(`${API_BASE}/ask`, {
        question,
        doc_ids: selectedDocs,
        sessionId,
      });
      setChatHistory((prev) => [
        ...prev,
        { role: "bot", text: res.data.answer },
      ]);
    } catch {
      setChatHistory((prev) => [
        ...prev,
        { role: "bot", text: "Error getting answer." },
      ]);
    }
    setAsking(false);
  };

  // ===============================
  // Summarize
  // ===============================
  const summarizePDF = async () => {
    if (selectedDocs.length === 0) return;
    setSummarizing(true);
    try {
      const res = await axios.post(`${API_BASE}/summarize`, {
        doc_ids: selectedDocs,
        sessionId,
      });
      setChatHistory((prev) => [
        ...prev,
        { role: "bot", text: res.data.summary },
      ]);
    } catch {
      alert("Error summarizing.");
    }
    setSummarizing(false);
  };

  // ===============================
  // Compare
  // ===============================
  const compareDocuments = async () => {
    if (selectedDocs.length < 2) return;
    setComparing(true);
    try {
      const res = await axios.post(`${API_BASE}/compare`, {
        doc_ids: selectedDocs,
        sessionId,
      });
      if (selectedDocs.length === 2) {
        setComparisonResult(res.data.comparison);
      } else {
        setChatHistory((prev) => [
          ...prev,
          { role: "user", text: "Compare selected documents." },
          { role: "bot", text: res.data.comparison },
        ]);
      }
    } catch {
      alert("Error comparing documents.");
    }
    setComparing(false);
  };

  const selectedPdfs = pdfs.filter((p) => selectedDocs.includes(p.doc_id));

  // ===============================
  // Theme classes
  // ===============================
  const pageBg = darkMode ? "bg-dark text-light" : "bg-light text-dark";

  const cardClass = darkMode
    ? "text-white border-secondary shadow"
    : "bg-white text-dark border-0 shadow-sm";

  const inputClass = darkMode
    ? "text-white border-secondary placeholder-white"
    : "";

  return (
    <div
      className={pageBg}
      style={{
        minHeight: "100vh",
        "--bs-card-bg": darkMode ? "#2c2c2c" : "#ffffff",
        "--bs-body-bg": darkMode ? "#1e1e1e" : "#f8f9fa",
      }}
    >
      {/* Navbar */}
      <Navbar
        bg={darkMode ? "dark" : "primary"}
        variant="dark"
        className="shadow mb-4 bg-gradient"
      >
        <Container className="d-flex justify-content-between align-items-center">
          <Navbar.Brand className="fw-bold d-flex align-items-center gap-2">
            <span role="img" aria-label="Bot">
              🤖
            </span>
            PDF Q&A Bot
          </Navbar.Brand>
          <div className="d-flex align-items-center gap-2">
            <span className="text-white small">
              {darkMode ? "⭐ Dark" : "🔆 Light"}
            </span>
            <div className="form-check form-switch mb-0">
              <input
                className="form-check-input"
                type="checkbox"
                role="switch"
                id="darkModeToggle"
                checked={darkMode}
                onChange={() => setDarkMode(!darkMode)}
                aria-label="Toggle dark/light mode"
                style={{ cursor: "pointer", width: "40px", height: "22px" }}
              />
            </div>
          </div>
        </Container>
      </Navbar>

      <Container className="mt-2">
        {/* Upload Card */}
        <Card className={`mb-4 ${cardClass}`}>
          <Card.Body>
            <Form>
              <Form.Control
                type="file"
                className={inputClass}
                onChange={(e) => setFile(e.target.files[0])}
              />
              <Button
                className="mt-2"
                onClick={uploadPDF}
                disabled={!file || uploading}
              >
                {uploading ? (
                  <Spinner size="sm" animation="border" />
                ) : (
                  "Upload"
                )}
              </Button>
            </Form>
          </Card.Body>
        </Card>

        {/* Selection Card */}
        {pdfs.length > 0 && (
          <Card className={`mb-4 ${cardClass}`}>
            <Card.Body>
              <h5>Select Documents</h5>
              {pdfs.map((pdf) => (
                <Form.Check
                  key={pdf.doc_id}
                  type="checkbox"
                  label={pdf.name}
                  checked={selectedDocs.includes(pdf.doc_id)}
                  onChange={() => toggleDocSelection(pdf.doc_id)}
                />
              ))}
            </Card.Body>
          </Card>
        )}

        {/* Side-by-side View */}
        {selectedPdfs.length === 2 && (
          <>
            <Row className="mb-4">
              {selectedPdfs.map((pdf) => (
                <Col key={pdf.doc_id} md={6}>
                  <Card className={cardClass}>
                    <Card.Body>
                      <h6>{pdf.name}</h6>
                      <Document file={pdf.url}>
                        <Page pageNumber={1} />
                      </Document>
                    </Card.Body>
                  </Card>
                </Col>
              ))}
            </Row>

            <Card className={`mb-4 ${cardClass}`}>
              <Card.Body>
                <Button
                  variant="info"
                  onClick={compareDocuments}
                  disabled={comparing}
                >
                  {comparing ? (
                    <Spinner size="sm" animation="border" />
                  ) : (
                    "Generate Comparison"
                  )}
                </Button>

                {comparisonResult && (
                  <div className="mt-4">
                    <h5>AI Comparison</h5>
                    <ReactMarkdown>{comparisonResult}</ReactMarkdown>
                  </div>
                )}
              </Card.Body>
            </Card>
          </>
        )}

        {/* Chat Mode */}
        {selectedPdfs.length !== 2 && (
          <Card className={cardClass}>
            <Card.Body>
              <h5>Ask Across Selected Documents</h5>
              <div
                style={{ maxHeight: 300, overflowY: "auto", marginBottom: 16 }}
              >
                {chatHistory.map((msg, i) => (
                  <div key={i} className="mb-2">
                    <strong>{msg.role === "user" ? "You" : "Bot"}:</strong>
                    <ReactMarkdown>{msg.text}</ReactMarkdown>
                  </div>
                ))}
              </div>

              <Form
                className="d-flex gap-2 mb-3"
                onSubmit={(e) => e.preventDefault()}
              >
                <Form.Control
                  type="text"
                  placeholder="Ask a question..."
                  className={inputClass}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      askQuestion();
                    }
                  }}
                />
                <Button
                  variant="success"
                  onClick={askQuestion}
                  disabled={asking}
                >
                  {asking ? <Spinner size="sm" animation="border" /> : "Ask"}
                </Button>
              </Form>

              <Button
                variant="warning"
                className="me-2"
                onClick={summarizePDF}
              >
                {summarizing ? (
                  <Spinner size="sm" animation="border" />
                ) : (
                  "Summarize"
                )}
              </Button>

              <Button
                variant="info"
                onClick={compareDocuments}
                disabled={selectedDocs.length < 2}
              >
                {comparing ? (
                  <Spinner size="sm" animation="border" />
                ) : (
                  "Compare Selected"
                )}
              </Button>
            </Card.Body>
          </Card>
        )}
      </Container>
    </div>
  );
}

export default App;
