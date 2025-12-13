import streamlit as st
import tempfile, os, sys
from pathlib import Path

# ---------------- Path setup ----------------
sys.path.insert(0, str(Path(__file__).parent))
from src.diagnosis_pipeline import run_pipeline_on_image
from src.config import CONFIDENCE_THRESHOLD


# ---------------- Page config ----------------
st.set_page_config(
    page_title="AI CT Scan Diagnosis",
    page_icon="🏥",
    layout="wide"
)


# ---------------- Utility: Image display ----------------
def display_image(file):
    st.markdown(
        """
        <style>
        .ct-image img {
            height: 50px;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # st.markdown('<div class="ct-image">', unsafe_allow_html=True)
    st.image(file,width = 300)
    # st.markdown('</div>', unsafe_allow_html=True)


# ---------------- Header ----------------
st.markdown("## 🏥 AI-Powered CT Scan Diagnosis")
st.caption("CNN-based image analysis with RAG-powered medical reports")
st.divider()


# ---------------- Session state ----------------
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "chat" not in st.session_state:
    st.session_state.chat = []


with st.sidebar:
    # -------- Navigation --------
    st.subheader("Navigation")
    mode = st.radio("", ["Image Analysis", "Chat Assistant"])
    st.divider()

    # -------- System Info --------
    st.subheader("System Info")
    st.markdown(
    """
    <style>
    .system-box {
        background-color: #111827;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #38bdf8;
        margin-top: 8px;
        color: white;
    }
    .system-box b { color: #38bdf8; }
    </style>

    <div class="system-box">
        <p><b>Model</b></p>
        <p>CNN: ResNet50</p>
        <p>Classes: Cancer, Tumor, Aneurysm</p>
        <p><b>LLM & Retrieval</b></p>
        <p>LLM: Llama-3.1 (Groq)</p>
        <p>RAG: Medical literature</p>
    </div>
    """,
    unsafe_allow_html=True
    )



    st.divider()

    # -------- Reset button --------
    if st.button("Reset"):
        st.session_state.last_result = None
        st.session_state.chat = []


# ================= IMAGE ANALYSIS =================
if mode == "Image Analysis":

    left, spacer, right = st.columns([1, 0.3, 0.5])

    # -------- Upload panel --------
    with left:
        st.subheader("Upload CT Image")

        file = st.file_uploader(
            "DICOM / JPG / PNG",
            type=["dcm", "jpg", "jpeg", "png"]
        )

        threshold = st.slider(
            "Confidence Threshold",
            0.0, 1.0,
            CONFIDENCE_THRESHOLD,
            0.05
        )

        run = st.button("Run Diagnosis", use_container_width=True)

    # -------- Image preview --------
    with right:
        st.subheader("Image Preview",text_alignment="center")
        if file:
            display_image(file)
        else:
            st.info("Upload an image to preview")

    # -------- Run inference --------
    if file and run:
        with st.spinner("Running analysis..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                tmp.write(file.read())
                path = tmp.name

            try:
                result = run_pipeline_on_image(path, threshold)
                st.session_state.last_result = result
            finally:
                os.remove(path)

    # -------- Results --------
    if st.session_state.last_result:
        res = st.session_state.last_result["prediction"]

        st.divider()
        st.subheader("Diagnosis Summary")
        with st.container(border=True):
            h1, h2, h3 = st.columns(3)

            with h1:
                st.caption("Detected Condition")
                st.markdown(f"### {res['disease']}")

            with h2:
                st.caption("Confidence")
                st.markdown(f"### {res['confidence']*100:.2f}%")

            with h3:
                st.caption("Model")
                st.markdown("### ResNet50")


        st.divider()
        st.subheader("Medical Report")

        with st.container(border=True):
            st.write(st.session_state.last_result["final_report"])

        if st.session_state.last_result.get("sources"):
            with st.expander("Sources"):
                for s in st.session_state.last_result["sources"]:
                    st.write(f"- {s.get('source')}")


# ================= CHAT ASSISTANT =================
else:
    st.subheader("Medical Chat Assistant")
    st.caption("Ask follow-up questions based on the diagnosis")

    for msg in st.session_state.chat:
        st.chat_message(msg["role"]).write(msg["content"])

    question = st.chat_input("Ask a medical question...")

    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                from src.retriever import build_rag_chain
                rag_chain, _ = build_rag_chain()

                context = ""
                if st.session_state.last_result:
                    p = st.session_state.last_result["prediction"]
                    context = f"Detected {p['disease']} with {p['confidence']:.2%} confidence.\n"

                answer = rag_chain.invoke(
                    {"input": context + question}
                ).get("answer", "Unable to answer.")

                st.write(answer)
                st.session_state.chat.append(
                    {"role": "assistant", "content": answer}
                )


# ---------------- Footer ----------------
st.divider()
st.caption(
    "For research and educational use only. "
    "Not a substitute for professional medical advice."
)
