# ── 📦 Import Required Libraries ─────────────────────────────
import os
import torch
import warnings
import pinecone
import streamlit as st
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# ── 🔐 Load Environment Variables ───────────────────────────
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Check if required API keys are present
if not PINECONE_API_KEY or not GOOGLE_API_KEY or not HUGGINGFACE_API_KEY:
    st.error("Missing API keys. Please check your .env configuration.")

# ── 🧠 Load Medical Embedding Model ─────────────────────────
MODEL_NAME = "abhinand/MedEmbed-large-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HUGGINGFACE_API_KEY)
model = AutoModel.from_pretrained(MODEL_NAME, token=HUGGINGFACE_API_KEY)

@st.cache_resource
def generate_embeddings(text):
    """Generate embeddings for a given text input using MedEmbed model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pooled = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return pooled.squeeze().tolist()

class CustomEmbeddings:
    """Wraps embedding generation for LangChain's vector store interface."""
    def embed_query(self, query: str):
        return generate_embeddings(query)

embedding_model = CustomEmbeddings()

# ── 🗃️ Initialize Pinecone Vector Store ─────────────────────
@st.cache_resource
def initialize_vectorstore():
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    index = pinecone_client.Index("personal-test-1")
    return PineconeVectorStore(
        index=index,
        embedding=embedding_model,
        text_key="text"
    )

# ── 💬 Initialize Gemini LLM ────────────────────────────────
@st.cache_resource
def load_llm():
    return GoogleGenerativeAI(
        model='gemini-2.0-flash-thinking-exp',
        api_key=GOOGLE_API_KEY
    )

# ── 🧾 Prompt Construction ──────────────────────────────────
def create_prompt():
    """Returns a ChatPromptTemplate with structured medical guidance."""
    system_prompt = """
You are a seasoned medical professor with deep expertise in clinical care and medical education. 
Your role is to provide accurate, structured, and easy-to-understand answers based on retrieved context from the USMLE dataset.

<context>
{context}
</context>

Follow these instructions depending on the query type:

1. **Single Concept Questions**
   - Define and explain the topic with structure: causes, symptoms, diagnosis, treatment, prognosis, etc.
   - Use examples, analogies, and a summary.

2. **Comparative Questions**
   - Clearly differentiate across aspects like cause, symptoms, diagnosis, and treatment.
   - Use tables and real-world examples.

3. **Multiple Choice (MCQ)**
   - Use a step-by-step approach ("Let's analyze...").
   - Break down patient case details and explain reasoning.
   - Analyze each option individually.
   - Highlight the correct answer with justification.

4. **Table Output**
   - Use Markdown table format.
   - Fill every row with complete data (use 'N/A' where not applicable).
   - Summarize and explain the table clearly.

Maintain an educational tone at all times. If the query lacks clarity, ask for more information before answering.

Questions: {input}
    """

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

# ── 🔄 Answer Generation Chain ───────────────────────────────
def answer_question(user_query):
    llm = load_llm()
    prompt = create_prompt()
    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever()
    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)
    response = retrieval_chain.invoke({"input": user_query})
    return response["answer"]

# ── 🖥️ Streamlit Interface ─────────────────────────────────
def main_interface():
    st.title("🧠 USMLE Step-1 Chatbot (personal-test-1)")
    user_query = st.text_area("Ask your USMLE Step-1 question:", height=100)
    if st.button("Get Answer", type="primary") and user_query:
        with st.spinner("Thinking..."):
            response = answer_question(user_query)
            st.write(response)

# ── ⚠️ Consent & Disclaimer ────────────────────────────────
if "consent_given" not in st.session_state:
    st.session_state.consent_given = False

if not st.session_state.consent_given:
    st.warning("⚠️ Please read and accept the disclaimer before using the chatbot.")
    st.write("""
- This chatbot is for **educational purposes only**.
- It does **not** provide medical advice or diagnoses.
- Always consult a licensed healthcare provider for medical concerns.
- In emergencies, seek immediate professional help.

:red[By clicking "I Agree", you acknowledge these terms.]
    """)
    if st.button("I Agree"):
        st.session_state.consent_given = True
else:
    main_interface()
