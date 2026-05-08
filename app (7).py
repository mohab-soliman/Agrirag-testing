import streamlit as st
import os
import base64
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ── تحميل اللوجو ─────────────────────────────────────────────────────────
def get_logo_base64(path="logo.png"):
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{data}"
    except:
        return None
    return None

logo_src = get_logo_base64()
logo_html = f'<img src="{logo_src}" style="height:80px; margin-bottom:10px;">' if logo_src else "🌱"

# ── إعدادات الصفحة ────────────────────────────────────────────────────────
st.set_page_config(page_title="AGRIRA - Intelligent Agriculture RAG", page_icon="🌱")

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');

#MainMenu, footer, header { visibility: hidden; }

html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Cairo', sans-serif !important;
    background-color: #f0f7f2 !important;
}

/* ── كل النصوص جوا الرسائل تبان ── */
[data-testid="stChatMessage"],
[data-testid="stChatMessage"] *,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] * {
    color: #1a1a1a !important;
    font-family: 'Cairo', sans-serif !important;
}

/* ── شكل الرسائل ── */
[data-testid="stChatMessage"] {
    background-color: #ffffff !important;
    border: 1px solid #c8e0d0 !important;
    border-radius: 16px !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.06) !important;
}

/* رسالة المستخدم */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background-color: #d6ede1 !important;
    border-left: 5px solid #1b4f31 !important;
}

/* رسالة الـ assistant */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background-color: #f8fffc !important;
    border-left: 5px solid #2b7a8a !important;
}

/* ── input ── */
[data-testid="stChatInput"] textarea {
    border-radius: 25px !important;
    border: 2px solid #a8d0bc !important;
    font-family: 'Cairo', sans-serif !important;
    font-size: 15px !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #1b4f31 !important;
    box-shadow: 0 0 0 3px rgba(27,79,49,0.1) !important;
}

/* ── هيدر ── */
.custom-header {
    background: linear-gradient(135deg, #1b4f31 0%, #2b7a8a 100%);
    padding: 24px 20px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 6px 20px rgba(27,79,49,0.25);
}
.custom-header h2 {
    margin: 8px 0 4px;
    font-size: 1.8rem;
    font-weight: 700;
    color: white !important;
    letter-spacing: 1px;
}
.custom-header p {
    margin: 0;
    font-size: 1rem;
    color: rgba(255,255,255,0.85) !important;
}

/* ── قائمة الترحيب ── */
.welcome-list-item {
    list-style: none;
    padding: 6px 0 6px 30px;
    position: relative;
    margin-bottom: 6px;
    font-size: 15px;
    color: #1a1a1a !important;
}
.welcome-list-item::before {
    content: "🌱";
    position: absolute;
    left: 0;
}

/* ── citation box ── */
.citation-box {
    background: #f4fbf7 !important;
    border-left: 4px solid #1b4f31;
    border-radius: 0 10px 10px 0;
    padding: 10px 14px;
    margin-top: 8px;
    font-size: 12px;
}
.citation-label {
    font-size: 11px;
    font-weight: 700;
    color: #1b4f31 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.citation-item {
    margin: 4px 0;
    color: #444 !important;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ── الهيدر ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="custom-header">
    {logo_html}
    <h2>AGRIRA</h2>
    <p>Intelligent Agriculture RAG 🌿</p>
</div>
""", unsafe_allow_html=True)

# ── رسالة الترحيب ─────────────────────────────────────────────────────────
with st.chat_message("assistant"):
    st.markdown("""
    Hello I am <b>AGRIRA</b><br>
    Your smart assistant in climate-smart agriculture 🌾<br><br>
    I can help you with:<br>
    <div class="welcome-list-item">Choosing suitable crops based on climate</div>
    <div class="welcome-list-item">Optimizing water consumption</div>
    <div class="welcome-list-item">Adapting to climate changes</div>
    <div class="welcome-list-item">Improving land productivity sustainably</div>
    """, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── APA Citation من الـ source (PyPDFLoader) ──────────────────────────────
def build_apa_citation(metadata):
    source = metadata.get("source", "")
    page   = metadata.get("page", None)
    if source:
        filename = os.path.basename(source)
        title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
    else:
        title = "Unknown Source"
    citation = f"{title}."
    if page is not None:
        try:
            citation += f" p. {int(page) + 1}"
        except:
            citation += f" p. {page}"
    return citation

# ── RAG Chain ─────────────────────────────────────────────────────────────
@st.cache_resource
def build_rag_chain():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vdb_path = os.path.join(current_dir, "VDB")

    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        encode_kwargs={"normalize_embeddings": True}
    )
    vector_store = Chroma(
        persist_directory=vdb_path,
        embedding_function=embedding_model
    )
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20}
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        convert_system_message_to_human=True
    )
    system_prompt = (
        "You are AGRIRA, a professional Agriculture Assistant. "
        "Use the retrieved context about agriculture to answer the user's question. "
        "If the answer is not in the context, say that you don't know. "
        "\n\nContext: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

rag_chain = build_rag_chain()

# ── عرض تاريخ المحادثة ────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("citations"):
            items = "".join(
                f'<div class="citation-item">📄 {c}</div>'
                for c in message["citations"]
            )
            st.markdown(
                f'<div class="citation-box">'
                f'<div class="citation-label">📚 References</div>'
                f'{items}</div>',
                unsafe_allow_html=True
            )

# ── الـ Query ─────────────────────────────────────────────────────────────
query = st.chat_input("Ask about agriculture topics...")
if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("AGRIRA is thinking..."):
        result = rag_chain.invoke({"input": query})
        answer = result["answer"]

        seen = set()
        citations_list = []
        for doc in result.get("context", []):
            citation = build_apa_citation(doc.metadata)
            if citation not in seen:
                seen.add(citation)
                citations_list.append(citation)

        with st.chat_message("assistant"):
            st.markdown(answer)
            if citations_list:
                items = "".join(
                    f'<div class="citation-item">📄 {c}</div>'
                    for c in citations_list
                )
                st.markdown(
                    f'<div class="citation-box">'
                    f'<div class="citation-label">📚 References</div>'
                    f'{items}</div>',
                    unsafe_allow_html=True
                )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations_list
    })
