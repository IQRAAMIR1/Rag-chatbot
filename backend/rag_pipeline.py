import os
import uuid
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

load_dotenv()

# Models initialize karo
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text(content: bytes, filename: str) -> str:
    if filename.endswith(".pdf"):
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    else:
        return content.decode("utf-8")

def process_document(content: bytes, filename: str) -> dict:
    # Text extract karo
    text = extract_text(content, filename)
    if not text.strip():
        return {"error": "Document mein text nahi mila"}

    # Chunks banao
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    # Unique doc ID banao
    doc_id = str(uuid.uuid4())[:8]

    # Collection banao ChromaDB mein
    collection = chroma_client.get_or_create_collection(name=f"doc_{doc_id}")

    # Embeddings banao aur store karo
    embeddings = embedding_model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks": len(chunks),
        "message": "Document successfully processed!"
    }

def query_document(question: str, doc_id: str) -> dict:
    try:
        collection = chroma_client.get_collection(name=f"doc_{doc_id}")
    except:
        return {"error": "Document nahi mila. Pehle upload karo."}

    # Question ka embedding banao
    q_embedding = embedding_model.encode([question]).tolist()

    # Similar chunks dhundo
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=5
    )
    context = "\n\n".join(results["documents"][0])

    # Groq se jawab lo
    prompt = f"""You are an intelligent document assistant. Your job is to answer questions based on the provided context from a document.

STRICT RULES:
- Answer ONLY in English
- Answer based on the context provided
- If the context has relevant information, use it to give a detailed helpful answer
- Only say "not available" if the context has absolutely NO relevant information
- Be conversational and helpful
- Do not be overly restrictive

Context from document:
{context}

User Question: {question}

Provide a clear, helpful answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": results["documents"][0][:2]
    }