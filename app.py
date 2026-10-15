import os
import streamlit as st
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("RAG Chatbot")

# 1. Pinecone & Τοπικά HuggingFace Embeddings (384 διαστάσεις)
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")  # π.χ. rag-minilm
index = pc.Index(index_name)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

embeddings = load_embeddings()
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# 2. Αρχικοποίηση LLM (Δωρεάν μέσω Groq)
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    api_key=os.environ.get("GROQ_API_KEY")
)

# 3. Chat History στο Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Εμφάνιση παλαιότερων μηνυμάτων
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# 4. Χειρισμός Ερώτησης Χρήστη
prompt = st.chat_input("Κάνε μια ερώτηση...")

if prompt:
    # Εμφάνιση και αποθήκευση της ερώτησης
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Ανάκτηση σχετικών αποσπασμάτων
    # Χρησιμοποιούμε similarity search (ή threshold 0.3-0.5)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    with st.spinner("Αναζήτηση πληροφοριών..."):
        docs = retriever.invoke(prompt)
        docs_text = "\n\n".join(d.page_content for d in docs)

    system_prompt = f"""You are a helpful assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the user's question.
If the answer cannot be found in the context, state clearly that you do not know.
Keep the answer concise and clear.

Context:
{docs_text}
"""

    # Δημιουργία payload μηνυμάτων (System prompt μόνο για το τρέχον query)
    conversation_payload = [
        SystemMessage(content=system_prompt),
        *st.session_state.messages
    ]

    with st.chat_message("assistant"):
        with st.spinner("Σκέφτομαι..."):
            response = llm.invoke(conversation_payload).content
            st.markdown(response)

    # Αποθήκευση της απάντησης στο ιστορικό
    st.session_state.messages.append(AIMessage(content=response))