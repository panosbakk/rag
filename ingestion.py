import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

REGISTRY_FILE = "processed_files.json"

def get_file_hash(filepath: str) -> str:
    """Υπολογίζει το SHA256 του PDF αρχείου."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_registry() -> dict:
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_registry(registry: dict):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

def load_pdf_pages(file_path: str):
    """Εξάγει κείμενο ανά σελίδα χωρίς το langchain-community."""
    reader = PdfReader(file_path)
    pages = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path, "page": idx + 1}
                )
            )
    return pages

# 1. Αρχικοποίηση Pinecone με διάσταση 384
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")

existing_indexes = [i["name"] for i in pc.list_indexes()]
if index_name not in existing_indexes:
    print(f"Δημιουργία index '{index_name}' με διάσταση 384...")
    pc.create_index(
        name=index_name,
        dimension=384,  # Διάσταση για το all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    print("Το index είναι έτοιμο.")

index = pc.Index(index_name)

# 2. Τοπικά Embeddings (Δωρεάν, τρέχει offline)
print("Φόρτωση τοπικού μοντέλου embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # 'cuda' αν έχεις κάρτα Nvidia
)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# 3. Φόρτωση και φιλτράρισμα αρχείων
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
registry = load_registry()
pdf_dir = Path("pdfpath/")  # Ο φάκελος με τα PDF σου

new_documents = []

for file_path in pdf_dir.glob("*.pdf"):
    str_path = str(file_path)
    file_hash = get_file_hash(str_path)

    if registry.get(str_path) == file_hash:
        print(f"Παράλειψη (ήδη ανεβασμένο): {file_path.name}")
        continue

    print(f"Επεξεργασία αρχείου: {file_path.name}")
    raw_docs = load_pdf_pages(str_path)
    split_docs = text_splitter.split_documents(raw_docs)
    new_documents.extend(split_docs)
    registry[str_path] = file_hash

# 4. Αποστολή στο Pinecone σε batches
if new_documents:
    BATCH_SIZE = 100
    for i in range(0, len(new_documents), BATCH_SIZE):
        batch = new_documents[i : i + BATCH_SIZE]
        batch_ids = [str(uuid.uuid4()) for _ in batch]
        vector_store.add_documents(documents=batch, ids=batch_ids)
        print(f"Ανέβηκαν {min(i + BATCH_SIZE, len(new_documents))} / {len(new_documents)} chunks.")

    save_registry(registry)
    print("Η διαδικασία ολοκληρώθηκε επιτυχώς!")
else:
    print("Δεν βρέθηκαν νέα αρχεία προς επεξεργασία.")