# RAG Document Assistant with Streamlit, Pinecone & Groq

Ένα πλήρες σύστημα **Retrieval-Augmented Generation (RAG)** με διαδραστικό UI σε **Streamlit**, ενσωματωμένο μηχανισμό δυναμικής εισαγωγής εγγράφων (Dynamic Ingestion) με αποφυγή διπλότυπων (deduplication μέσω SHA256 hashes), vector database στο **Pinecone** και LLM inference υπερυψηλής ταχύτητας μέσω **Groq API** με το μοντέλο **`openai/gpt-oss-120b`**.

Το project είναι πλήρως containerized με **Docker Compose**, εξασφαλίζοντας επίμονη αποθήκευση (persistence) δεδομένων στον κεντρικό φάκελο και χαμηλή κατανάλωση πόρων (CPU-only PyTorch build).

---

## Αρχιτεκτονική & Χαρακτηριστικά

* **Διαδραστικό UI & Ingestion (`app.py`):** Περιβάλλον Streamlit με Drag-and-Drop ανέβασμα PDF αρχείων, real-time πρόοδο, text chunking και άμεσο upsert στο Pinecone.
* **Persistent Registry (`processed_files.json`):** Υπολογισμός SHA256 checksum για κάθε έγγραφο με atomic disk write και `fsync`, αποτρέποντας διπλοεγγραφές και απώλεια καταγραφής μετά από επανεκκίνηση.
* **Embeddings Model:** Τοπική παραγωγή διανυσμάτων μέσω του μοντέλου `sentence-transformers/all-MiniLM-L6-v2` (384 διαστάσεις) σε CPU.
* **Vector Database:** Pinecone Serverless Index (Cosine similarity).
* **LLM Engine:** Groq API με το μοντέλο **`openai/gpt-oss-120b`** για παραγωγή απαντήσεων βασισμένων αυστηρά στο ανακτηθέν πλαίσιο (context).
* **Reset Feature:** Ενσωματωμένη δυνατότητα πλήρους εκκαθάρισης της βάσης διανυσμάτων και του registry απευθείας από την πλαϊνή μπάρα.

---

## Δομή του Project


├── app.py                   # Κεντρική εφαρμογή Streamlit (Chatbot & UI Ingestion)
├── ingestion.py             # CLI script για offline batch ingestion (έχει μείνει μόνο για την περίπτωση ανεβάσματος μεγάλου αρχείου, ως backup για να μην κολλήσει το streamlit)
├── processed_files.json     # Μητρώο επεξεργασμένων αρχείων (SHA256 hashes)
├── requirements.txt         # Εξαρτήσεις βιβλιοθηκών Python
├── Dockerfile               # Multi-stage / CPU-optimized Docker image
├── docker-compose.yml       # Ενορχήστρωση υπηρεσιών και persistent volumes
├── .dockerignore            # Αποκλεισμός περιττών αρχείων από το build context
├── .env.example             # Πρότυπο περιβαλλοντικών μεταβλητών
└── README.md                # Τεκμηρίωση του project

--- 

## Οδηγίες Χρήσης & Διαχείριση Docker

### 1. Ρύθμιση Περιβάλλοντος (`.env`)
Δημιουργήστε ένα αρχείο `.env` στον κεντρικό φάκελο του project (βασισμένο στο `.env.example`):
```bash
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_index_name_here
GROQ_API_KEY=your_groq_api_key_here
```
---

### 2. Build and Run
Στον φάκελο του πρότζεκτ:
docker compose up -d --build (αν τρέχει την πρώτη φορά και χρειάζεται το build)
docker compose up -d (αν έχει προηγηθεί το build)
Για να σταματήσει: docker compose down.

---

## Παραδείγματα

