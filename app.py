import hashlib
import json
import os
import uuid
import streamlit as st
from dotenv import load_dotenv
import tempfile
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")


# --- Βοηθητικές Συναρτήσεις Ingestion ---
def get_bytes_hash(file_bytes: bytes) -> str:
    """Υπολογίζει το SHA256 των bytes του αρχείου."""
    return hashlib.sha256(file_bytes).hexdigest()


REGISTRY_FILE = "processed_files.json"

def load_registry() -> dict:
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading registry: {e}")
            return {}
    return {}

def save_registry(registry: dict):
    dirname = os.path.dirname(os.path.abspath(REGISTRY_FILE))
    try:
        with tempfile.NamedTemporaryFile("w", dir=dirname, delete=False, encoding="utf-8") as tf:
            json.dump(registry, tf, indent=2)
            tf.flush()
            os.fsync(tf.fileno())
            temp_name = tf.name
        
        os.replace(temp_name, REGISTRY_FILE)
    except Exception as e:
        print(f"Error saving registry: {e}")
        if 'temp_name' in locals() and os.path.exists(temp_name):
            os.remove(temp_name)

def extract_pdf_pages_from_file(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": uploaded_file.name, "page": idx + 1}
                )
            )
    return pages


pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

embeddings = load_embeddings()
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    api_key=os.environ.get("GROQ_API_KEY")
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


with st.sidebar:
    st.header("📄 Διαχείριση Εγγράφων")
    uploaded_files = st.file_uploader(
        "Σύρε ή επίλεξε PDF αρχεία",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Επεξεργασία & Ανέβασμα"):
            registry = load_registry()
            new_chunks = []
            
            with st.spinner("Ανάλυση αρχείων..."):
                for uploaded_file in uploaded_files:
                    file_bytes = uploaded_file.getvalue()
                    file_hash = get_bytes_hash(file_bytes)
                    
                    if registry.get(uploaded_file.name) == file_hash:
                        st.info(f"Το '{uploaded_file.name}' υπάρχει ήδη.")
                        continue
                    
                    raw_docs = extract_pdf_pages_from_file(uploaded_file)
                    split_docs = text_splitter.split_documents(raw_docs)
                    new_chunks.extend(split_docs)
                    registry[uploaded_file.name] = file_hash

                if new_chunks:
                    BATCH_SIZE = 100
                    progress_bar = st.progress(0)
                    for i in range(0, len(new_chunks), BATCH_SIZE):
                        batch = new_chunks[i : i + BATCH_SIZE]
                        batch_ids = [str(uuid.uuid4()) for _ in batch]
                        vector_store.add_documents(documents=batch, ids=batch_ids)
                        progress_bar.progress(min((i + BATCH_SIZE) / len(new_chunks), 1.0))
                    
                    save_registry(registry)
                    st.success(f"Επιτυχία! Προστέθηκαν {len(new_chunks)} νέα αποσπάσματα.")
                else:
                    st.warning("Δεν εντοπίστηκαν νέα έγγραφα προς αποθήκευση.")

    st.divider()
    st.subheader("Αρχεία στη βάση:")
    current_registry = load_registry()
    if current_registry:
        for fname in current_registry.keys():
            st.text(f"• {fname}")
    else:
        st.caption("Καμία καταχώρηση ακόμα.")
    st.divider()
    if st.button("🗑️ Εκκαθάριση Βάσης (Reset)", type="primary"):
        with st.spinner("Διαγραφή δεδομένων από το Pinecone..."):
            try:
                index.delete(delete_all=True)
                if os.path.exists(REGISTRY_FILE):
                    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
                        f.write("{}")
                st.session_state.messages = []
                st.success("Η βάση και το ιστορικό καθαρίστηκαν!")
                st.rerun()
            except Exception as e:
                st.error(f"Σφάλμα κατά τη διαγραφή: {e}")
# --- Chat Interface ---
st.title("💬 RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

prompt = st.chat_input("Κάνε μια ερώτηση για τα έγγραφά σου...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    with st.spinner("Αναζήτηση στα έγγραφα..."):
        docs = retriever.invoke(prompt)
        docs_text = "\n\n".join(d.page_content for d in docs)

    system_prompt = f"""You are a helpful assistant answering questions based on provided documents.
Use the following context to answer the question concisely and accurately.
If the answer is not contained in the context, say that you don't know.

Context:
{docs_text}
"""

    conversation_payload = [
        SystemMessage(content=system_prompt),
        *st.session_state.messages
    ]

    with st.chat_message("assistant"):
        with st.spinner("Σύνταξη απάντησης..."):
            response = llm.invoke(conversation_payload).content
            st.markdown(response)
            if docs:
                with st.expander("🔍 Σχετικές πηγές"):
                    for d in docs:
                        src = d.metadata.get("source", "Άγνωστο")
                        page = d.metadata.get("page", "-")
                        st.caption(f"**Αρχείο:** {src} (Σελίδα {page})")

    st.session_state.messages.append(AIMessage(content=response))