import React, { useState, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import "bootstrap/dist/css/bootstrap.min.css";
import {
  Container,
  Button,
  Form,
  Card,
  Spinner,
  Navbar,
} from "react-bootstrap";

const API_BASE = process.env.REACT_APP_API_URL || "";
const THEME_STORAGE_KEY = "pdf-qa-bot-theme";

function App() {
  // Core state
  const [file, setFile] = useState(null);
  const [pdfs, setPdfs] = useState([]); // {name, sessionId, url}
  const [selectedDocs, setSelectedDocs] = useState([]);
  const [question, setQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [comparisonResult, setComparisonResult] = useState(null);

  // UI state
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [summarizing, setSummarizing] = useState(false);
  const [comparing, setComparing] = useState(false);

  // Theme
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    return saved ? JSON.parse(saved) : false;
  });

  useEffect(() => {
    localStorage.setItem(THEME_STORAGE_KEY, JSON.stringify(darkMode));
  }, [darkMode]);

  // Upload
  const uploadDocument = async () => {
    if (!file) return;

    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await axios.post(`${API_BASE}/upload`, formData);

      const url = URL.createObjectURL(file);
      const serverSessionId = res.data?.sessionId; // Get sessionId from server

      setPdfs((prev) => [
        ...prev,
        { 
          name: file.name, 
          sessionId: serverSessionId, // Store server-returned sessionId
          url 
        },
      ]);

      setFile(null);
      alert("Document uploaded!");
    } catch {
      alert("Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  // Toggle docs
  const toggleDocSelection = (sessionId) => {
    setComparisonResult(null);
    setSelectedDocs((prev) =>
      prev.includes(sessionId)
        ? prev.filter((id) => id !== sessionId)
        : [...prev, sessionId]
    );
  };

  // Ask question
  const askQuestion = async () => {
    if (!question.trim() || selectedDocs.length === 0) return;

    setAsking(true);
    setChatHistory((prev) => [...prev, { role: "user", text: question }]);

    try {
      // Use the first selected document's sessionId
      const res = await axios.post(`${API_BASE}/ask`, {
        question,
        sessionId: selectedDocs[0], // Send the stored sessionId
      });

      setChatHistory((prev) => [
        ...prev,
        {
          role: "bot",
          text: res.data.answer,
        },
      ]);
    } catch {
      setChatHistory((prev) => [
        ...prev,
        { role: "bot", text: "Error getting answer." },
      ]);
    } finally {
      setQuestion("");
      setAsking(false);
    }
  };

  // Summarize
  const summarizePDF = async () => {
    if (selectedDocs.length === 0) return;

    setSummarizing(true);
    try {
      // Use the first selected document's sessionId
      const res = await axios.post(`${API_BASE}/summarize`, {
        sessionId: selectedDocs[0], // Send the stored sessionId
      });

      setChatHistory((prev) => [
        ...prev,
        { role: "bot", text: res.data.summary },
      ]);
    } catch {
      alert("Error summarizing.");
    } finally {
      setSummarizing(false);
    }
  };

  // Compare
  const compareDocuments = async () => {
    if (selectedDocs.length < 2) return;

    setComparing(true);
    try {
      const res = await axios.post(`${API_BASE}/compare`, {
        sessionId,
        doc_ids: selectedDocs,
      });

      setComparisonResult(res.data.comparison);

      setChatHistory((prev) => [
        ...prev,
        { role: "bot", text: res.data.comparison },
      ]);
    } catch {
      alert("Error comparing.");
    } finally {
      setComparing(false);
    }
  };

  const selectedPdfs = pdfs.filter((p) =>
    selectedDocs.includes(p.sessionId)
  );

  return (
    <div className={darkMode ? "bg-dark text-light" : "bg-light"}>
      <Navbar bg="primary" variant="dark">
        <Container className="d-flex justify-content-between">
          <Navbar.Brand>PDF Q&A Bot</Navbar.Brand>
          <Button onClick={() => setDarkMode(!darkMode)}>
            {darkMode ? "Light" : "Dark"}
          </Button>
        </Container>
      </Navbar>

      <Container className="mt-4">
        <Card className="mb-4">
          <Card.Body>
            <Form.Control
              type="file"
              accept=".pdf,.docx,.txt"
              onChange={(e) => setFile(e.target.files[0])}
            />
            <Button
              className="mt-2"
              onClick={uploadDocument}
              disabled={uploading}
            >
              {uploading ? <Spinner size="sm" /> : "Upload"}
            </Button>
          </Card.Body>
        </Card>

        {pdfs.map((pdf) => (
          <Form.Check
            key={pdf.sessionId}
            label={pdf.name}
            checked={selectedDocs.includes(pdf.sessionId)}
            onChange={() => toggleDocSelection(pdf.sessionId)}
          />
        ))}

        {selectedPdfs.length === 2 && (
          <Button onClick={compareDocuments} disabled={comparing}>
            {comparing ? "Comparing..." : "Compare"}
          </Button>
        )}

        <div style={{ maxHeight: 300, overflowY: "auto" }}>
          {chatHistory.map((msg, i) => (
            <div key={i}>
              <b>{msg.role}:</b>
              <ReactMarkdown>{msg.text}</ReactMarkdown>
            </div>
          ))}
        </div>

        <Form
          onSubmit={(e) => {
            e.preventDefault();
            askQuestion();
          }}
        >
          <Form.Control
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask..."
          />
        </Form>

        <Button onClick={summarizePDF} disabled={summarizing}>
          {summarizing ? "Summarizing..." : "Summarize"}
        </Button>
      </Container>
    </div>
  );
}

export default App;