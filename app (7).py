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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Cairo', sans-serif;
        background-color: #f2f8f5;
    }

    .stChatMessage {
        background-color: #fafdfb !important;
        border: 1px solid #e0ebe4 !important;
        border-radius: 15px !important;
        padding: 15px !important;
        margin-bottom: 10px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03) !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #d4ede0 !important;
        border-left: 5px solid #1b4f31 !important;
    }

    .stChatInputContainer {
        padding-bottom: 20px !important;
    }

    .custom-header {
        background: linear-gradient(135deg, #1b4f31 0%, #2b7a8a 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(27, 79, 49, 0.2);
    }
    .custom-header h2 { margin: 0; font-size: 1.5rem; }
    .custom-header p  { margin: 5px 0 0; font-size: 0.9rem; opacity: 0.9; }

    .welcome-list-item {
        list-style: none;
        padding-left: 25px;
        position: relative;
        margin-bottom: 10px;
    }
    .welcome-list-item::before { content: "🌱"; position: absolute; left: 0; }
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

# ── Citation ─────────────────────────────────────────────────────────────
def build_apa_citation(metadata):
    author = metadata.get("author", "")
    title  = metadata.get("title", "")
    year   = metadata.get("creationdate", "")
    page   = metadata.get("page", None)
    source = metadata.get("source", "")

    if year and len(year) >= 4:
        year = year[:4]
    else:
        year = "n.d."

    if not author:
        author = "Unknown Author"

    if not title:
        filename = os.path.basename(source)
        title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")

    citation = f"{author}. ({year}). {title}."
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
        model_name="intfloat/multilingual-e5-large"
    )
    vector_store = Chroma(
        persist_directory=vdb_path,
        embedding_function=embedding_model
    )
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        convert_system_message_to_human=True
    )
    system_prompt = (
        "You are AGRIRA, a professional Agriculture Assistant. "
        "Use the following retrieved documents to answer the user's question. "
        "Answer directly and helpfully based on the context. "
        "Do not say you don't know if the context contains relevant information. "
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
            st.markdown("---")
            st.markdown("<small>📚 <b>References</b></small>", unsafe_allow_html=True)
            for c in message["citations"]:
                st.caption(c)

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
                st.markdown("---")
                st.markdown("<small>📚 <b>References</b></small>", unsafe_allow_html=True)
                for c in citations_list:
                    st.caption(c)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations_list
    })
