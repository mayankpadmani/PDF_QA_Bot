from fastapi import FastAPI
from fastapi import Request
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import re
import uvicorn
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from slowapi import Limiter
from slowapi.util import get_remote_address

load_dotenv()

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Temporary global variables
vectorstore = None
qa_chain = False
HF_GENERATION_MODEL = os.getenv("HF_GENERATION_MODEL", "google/flan-t5-base")
generation_tokenizer = None
generation_model = None
generation_is_encoder_decoder = False

# Load local embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# TEXT NORMALIZATION UTILITIES
# ---------------------------------------------------------------------------

def normalize_spaced_text(text: str) -> str:
    """
    Fixes character-level spaced text produced by PyPDFLoader on certain
    vector-based PDFs (e.g. NPTEL / IBM Coursera certificates).

    Examples:
        'J A I N I   S O L A N K I'  ->  'JAINI SOLANKI'
        'I B M'                       ->  'IBM'
        'N P T E L'                   ->  'NPTEL'

    Normal multi-letter words are left completely untouched.
    """
    def fix_spaced_word(match):
        return match.group(0).replace(" ", "")

    # Pattern: 3+ single alpha chars each separated by exactly one space
    pattern = r'\b(?:[A-Za-z] ){2,}[A-Za-z]\b'
    return re.sub(pattern, fix_spaced_word, text)


def normalize_answer(text: str) -> str:
    """
    Post-processes the LLM-generated answer:
    - Removes any residual character-level spacing.
    - Strips prompt leakage (lines starting with 'Answer', 'Context', etc.)
    - Collapses excessive whitespace.
    """
    # Remove residual character spacing in the answer itself
    text = normalize_spaced_text(text)
    # Strip any prompt-leakage prefixes the model might echo
    text = re.sub(r'^(Answer[^:]*:|Context:|Question:)\s*', '', text, flags=re.IGNORECASE)
    # Collapse multiple spaces/newlines
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# MODEL LOADING & GENERATION
# ---------------------------------------------------------------------------

def load_generation_model():
    global generation_tokenizer, generation_model, generation_is_encoder_decoder
    if generation_model is not None and generation_tokenizer is not None:
        return generation_tokenizer, generation_model, generation_is_encoder_decoder

    config = AutoConfig.from_pretrained(HF_GENERATION_MODEL)
    generation_is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
    generation_tokenizer = AutoTokenizer.from_pretrained(HF_GENERATION_MODEL)

    if generation_is_encoder_decoder:
        generation_model = AutoModelForSeq2SeqLM.from_pretrained(
            HF_GENERATION_MODEL,
            low_cpu_mem_usage=False
        )
    else:
        generation_model = AutoModelForCausalLM.from_pretrained(
            HF_GENERATION_MODEL,
            low_cpu_mem_usage=False
        )

    if torch.cuda.is_available():
        generation_model = generation_model.to("cuda")

    generation_model.eval()
    return generation_tokenizer, generation_model, generation_is_encoder_decoder


def generate_response(prompt: str, max_new_tokens: int) -> str:
    tokenizer, model, is_encoder_decoder = load_generation_model()
    model_device = next(model.parameters()).device

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    encoded = {key: value.to(model_device) for key, value in encoded.items()}
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )

    if is_encoder_decoder:
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return text.strip()

    input_len = encoded["input_ids"].shape[1]
    new_tokens = generated_ids[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


# ---------------------------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------------------------

class PDFPath(BaseModel):
    filePath: str

class AskRequest(BaseModel):
    question: str
    history: list = []


class SummarizeRequest(BaseModel):
    pdf: str | None = None


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.post("/process-pdf")
@limiter.limit("15/15 minutes")
def process_pdf(request: Request, data: PDFPath):
    global vectorstore, qa_chain

    loader = PyPDFLoader(data.filePath)
    raw_docs = loader.load()

    # ── Layer 1: normalize at ingestion ──────────────────────────────────────
    # Clean each page's text before chunking so embeddings are on real words.
    cleaned_docs = []
    for doc in raw_docs:
        cleaned_content = normalize_spaced_text(doc.page_content)
        cleaned_docs.append(Document(page_content=cleaned_content, metadata=doc.metadata))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(cleaned_docs)
    if not chunks:
        return {"error": "No text chunks generated from the PDF. Please check your file."}

    vectorstore = FAISS.from_documents(chunks, embedding_model)
    qa_chain = True

    return {"message": "PDF processed successfully"}


@app.post("/ask")
@limiter.limit("60/15 minutes")
def ask_question(request: Request, data: AskRequest):
    global vectorstore, qa_chain
    if not qa_chain:
        return {"answer": "Please upload a PDF first!"}
    question = data.question
    history = data.history
    conversation_context = ""
    # Use only last 5 messages to avoid long prompts
    for msg in history[-5:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        conversation_context += f"{role}: {content}\n"
    docs = vectorstore.similarity_search(question, k=4)
    if not docs:
        return {"answer": "No relevant context found."}

    # ── Layer 2a: context is already clean (normalized at ingestion) ──────────
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a helpful assistant answering questions from a PDF document.

    Conversation History:
    {conversation_context}

    Document Context:
    {context}

    Current Question:
    {question}

    Instructions:
    - Use the document context to answer.
    - If the answer is not in the document, say so briefly.
    - Keep the answer concise.

    Answer:
    """

    raw_answer = generate_response(prompt, max_new_tokens=128)

    # ── Layer 3: post-process the answer itself ───────────────────────────────
    answer = normalize_answer(raw_answer)
    return {"answer": answer}


@app.post("/summarize")
@limiter.limit("15/15 minutes")
def summarize_pdf(request: Request, data: SummarizeRequest):
    global vectorstore, qa_chain
    if not qa_chain:
        return {"summary": "Please upload a PDF first!"}

    docs = vectorstore.similarity_search("Give a concise summary of the document.", k=6)
    if not docs:
        return {"summary": "No document context available to summarize."}

    # Context is already clean (normalized at ingestion)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = (
        "You are a document summarization assistant working with a certificate or official document.\n"
        "RULES:\n"
        "1. Summarize in 6-8 concise bullet points.\n"
        "2. Clearly distinguish: who received the certificate, what course, which company issued it,\n"
        "   who signed it, on what platform, and on what date.\n"
        "3. Return clean, properly formatted text — no character spacing, proper Title Case for names.\n"
        "4. Use ONLY the information in the context below.\n\n"
        f"Context:\n{context}\n\n"
        "Summary (bullet points):"
    )

    raw_summary = generate_response(prompt, max_new_tokens=256)
    summary = normalize_answer(raw_summary)
    return {"summary": summary}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
