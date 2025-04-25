import os
import requests
import json
import pdfplumber
import pinecone
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g. "us-west2-aws"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = "iq-bot-demo2"

# Debug prints (optional)
print("PINECONE_API_KEY:", bool(PINECONE_API_KEY))
print("PINECONE_ENV:", bool(PINECONE_ENV))
print("GOOGLE_API_KEY:", bool(GOOGLE_API_KEY))

# Safety check
assert PINECONE_API_KEY and PINECONE_ENV and GOOGLE_API_KEY, "❌ Missing environment variables. Check your .env file."

# === Step 0: Create Pinecone Index if not exists ===
def create_pinecone_index():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=768,  # Gemini embedding dimension
            metric="cosine"
        )
        print(f"✅ Index '{INDEX_NAME}' created.")
    else:
        print(f"ℹ️ Index '{INDEX_NAME}' already exists.")

# === Step 1: Extract PDF text ===
def extract_text_from_pdf(file_path):
    full_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    return full_text

# === Step 2: Chunk text ===
def chunk_text(text, chunk_size=1000, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# === Step 3: Get embeddings from Gemini ===
def get_embedding(text):
    api_key = os.getenv("GOOGLE_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "models/embedding-001",
        "content": {
            "parts": [
                {
                    "text": text
                }
            ]
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["embedding"]
    except Exception as e:
        print("❌ Embedding error:", e)
        return None

# === Step 4: Upsert to Pinecone without SDK ===
def upsert_to_pinecone(vectors):
    url = f"https://{INDEX_NAME}-{PINECONE_ENV}.svc.pinecone.io/vectors/upsert"
    headers = {
        "Content-Type": "application/json",
        "Api-Key": PINECONE_API_KEY
    }

    payload = {
        "vectors": vectors,
        "namespace": ""  # optional
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print(f"✅ Upserted {len(vectors)} vectors.")
    except Exception as e:
        print("❌ Upsert error:", e)

# === Step 5: Main function ===
def process_pdf_folder(folder_path):
    for file in os.listdir(folder_path):
        if not file.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(folder_path, file)
        print(f"📄 Processing {file}...")
        raw_text = extract_text_from_pdf(file_path)
        chunks = chunk_text(raw_text)

        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            if embedding:
                vector_id = f"{file}-{i}"
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk[:500]  # Optional metadata
                    }
                })

        if vectors:
            upsert_to_pinecone(vectors)

# === Entry point ===
if __name__ == "__main__":
    create_pinecone_index()  # 👈 Ensure index exists before processing
    pdf_folder = os.path.join(os.getcwd(), "IQ_TechMax")
    process_pdf_folder(pdf_folder)
    print("✅ All PDFs processed and uploaded to Pinecone.")
