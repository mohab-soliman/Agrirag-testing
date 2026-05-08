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
# FIX 1: مسار اللوجو بقى نسبي (logo.png في نفس الفولدر) بدل المسار اللوكال
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

    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #eef7f1 !important;
        border-left: 5px solid #1b4f31 !important;
    }

    .stChatInputContainer {
        padding-bottom: 20px !important;
    }
    .stChatInput input {
        border-radius: 25px !important;
        border: 1px solid #c2d6cb !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #d4ede0 !important;
        border-left: 5px solid #1b4f31 !important;
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
    .custom-header h2 {
        margin: 0;
        font-size: 1.5rem;
    }
    .custom-header p {
        margin: 5px 0 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }

    .welcome-list-item {
        list-style: none;
        padding-left: 25px;
        position: relative;
        margin-bottom: 10px;
    }
    .welcome-list-item::before {
        content: "🌱";
        position: absolute;
        left: 0;
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

def build_apa_citation(metadata):
    author = metadata.get("author", "Unknown Author")
    year = metadata.get("year", "n.d.")
    title = metadata.get("title", "Untitled")
    page = metadata.get("page", None)
    citation = f"{author}. ({year}). {title}."
    if page:
        citation += f" p. {int(page) + 1}"
    return citation

@st.cache_resource
def build_rag_chain():
    # FIX 2: مسار الـ VDB نسبي بيشتغل على Streamlit Cloud
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vdb_path = os.path.join(current_dir, "VDB")

    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vector_store = Chroma(
        persist_directory=vdb_path,
        embedding_function=embedding_model
    )
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    # FIX 3: الـ API Key بتيجي من st.secrets بس (مش هاردكودد)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

    system_prompt = (
        "You are AGRIRA, a professional Agriculture Assistant. "
        "Use the retrieved context about agriculture to answer the user's question. "
        "If the answer is not in the context, say that you don't know. "
        "\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

rag_chain = build_rag_chain()

# عرض تاريخ المحادثة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask about agriculture topics...")
if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # FIX 4: الـ chat_message جوا الـ spinner عشان الـ citations تظهر
    with st.spinner("AGRIRA is thinking..."):
        result = rag_chain.invoke({"input": query})
        answer = result["answer"]

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.markdown("---")
            st.markdown("<small>📚 <b>References</b></small>", unsafe_allow_html=True)
            seen_citations = set()
            for doc in result["context"]:
                citation = build_apa_citation(doc.metadata)
                if citation not in seen_citations:
                    seen_citations.add(citation)
                    st.caption(citation)

    # FIX 5: الـ append بعد العرض عشان مش يتكرر في الـ history
    st.session_state.messages.append({"role": "assistant", "content": answer})
