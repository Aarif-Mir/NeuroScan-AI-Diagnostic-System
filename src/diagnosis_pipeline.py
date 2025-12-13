"""
Orchestrator for:
DICOM → CNN Prediction → RAG → Medical Report

- Uses LangChain LCEL RAG via build_rag_chain()
- Automatically ensures PDFs are embedded
- Strong anti-hallucination constraints
"""

from typing import Dict, Any
import os
import numpy as np

from .inference import predict_image
from .retriever import build_rag_chain
from .rag_ingest import ingest_pdfs
from .config import (
    CHROMA_DIR,
    PDF_DIR,
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    LABEL_MAP,
)

# Lazy-loaded globals


_model_cache = None
_rag_chain = None

# Helpers


def _load_model():
    """Lazy-load the Keras CNN model."""
    global _model_cache
    if _model_cache is None:
        from tensorflow.keras.models import load_model
        model_path_str = str(MODEL_PATH)
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {model_path_str}")
        _model_cache = load_model(model_path_str)
    return _model_cache

def _ensure_chroma_db():
    """Check if Chroma DB exists; if not, ingest PDFs."""
    chroma_dir_str = str(CHROMA_DIR)
    pdf_dir_str = str(PDF_DIR)
    
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        print("Chroma DB missing or empty. Running PDF ingestion...")
        ingest_pdfs(pdf_dir=pdf_dir_str, persist_dir=chroma_dir_str)
    else:
        print("Chroma DB found. Skipping ingestion.")

def get_rag_chain():
    """Lazy-load RAG chain with persistent Chroma DB."""
    global _rag_chain
    if _rag_chain is None:
        _ensure_chroma_db()
        _rag_chain, _ = build_rag_chain()
    return _rag_chain

def build_report(disease: str, confidence: float, rag_answer: str, sources: list = None) -> str:
    confidence_pct = confidence * 100

    if confidence >= 0.8:
        conf_level = "Very High"
    elif confidence >= 0.6:
        conf_level = "High"
    elif confidence >= 0.4:
        conf_level = "Moderate"
    else:
        conf_level = "Low"

    sources_md = ""
    if sources:
        sources_md += "\n### 📚 Source Citations\n"
        for i, src in enumerate(sources, 1):
            sources_md += f"- {src.get('source', 'Unknown')} (Page {src.get('page', 'N/A')})\n"

    return f"""
##  AI-Assisted CT Scan Diagnostic Report

###  Model Prediction
- **Detected Condition:** {disease}
- **Confidence Score:** {confidence:.4f} ({confidence_pct:.1f}%)  
- **Confidence Level:** {conf_level}
- **Analysis Method:** Deep Learning CNN (ResNet-based)

---

###  Evidence-Based Medical Summary
{rag_answer}

{sources_md}

---

###  Clinical Notes
- Generated using **CNN + Retrieval-Augmented Generation (RAG)**
- Evidence sourced from peer-reviewed medical literature
- Accuracy depends on image quality and available literature

---

###  Important Disclaimer
This report is for **research and educational purposes only**.  
It is **NOT** a substitute for professional medical diagnosis.

Always consult a qualified **radiologist or physician** for clinical decisions.
""".strip()


# Main pipeline


def run_pipeline_on_image(image_path: str, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> Dict[str, Any]:
    """
    Full pipeline:
    1) Supports:
        - DICOM (.dcm)
        - Standard images (.jpg, .png, .jpeg)
    2) Map class → disease
    3) RAG-based evidence extraction
    4) Structured report
    """

    # Step 1: CNN prediction
    model = _load_model()
    class_id, confidence = predict_image(model=model, image_path=image_path)
    class_id = int(class_id)
    confidence = float(confidence)  # keep exact float

    disease = LABEL_MAP.get(class_id, "Unknown")
    is_confident = confidence >= confidence_threshold

    # Step 2: RAG query
    rag_chain = get_rag_chain()
    if is_confident:
        query = f"""## CT SCAN ANALYSIS: {disease} (Confidence: {confidence})

Task: Provide evidence-based medical information about {disease} using ONLY the provided context documents.

Required Sections:
1. Disease Overview: Definition, pathophysiology, epidemiology, clinical presentation
2. CT Imaging Features: Typical appearance, radiological patterns, distinguishing characteristics, enhancement patterns
3. Treatment Protocols: First-line/alternative treatments, efficacy, indications/contraindications
4. Diagnostic Workup: Additional imaging (MRI/PET), lab tests, biopsy, follow-up protocols
5. Clinical Management: Monitoring, prognosis, clinical considerations, long-term strategies
6. Prevention & Supportive Care: Risk reduction, lifestyle modifications, dietary recommendations

Requirements:
- Cite EVERY claim: [Source: filename.pdf, Page: X]
- Use ONLY information from provided documents (no general knowledge)
- If info missing, state: "Information not found in provided documents for [section]"
- Format: Markdown headers (##), bullet points, professional medical style
- Be comprehensive but concise"""
    else:
        query = f"""## CT SCAN ANALYSIS: UNCERTAIN PREDICTION - {disease} (Confidence: {confidence:.1%}, below {confidence_threshold})

Task: Provide differential diagnosis analysis and conservative clinical guidance using ONLY the provided context documents.

Required Sections:
1. Differential Diagnosis: List 3-5+ conditions with similar CT findings (include {disease}), rank by likelihood, describe distinguishing features
2. Supporting Evidence: For EACH differential, provide CT imaging characteristics, clinical features, and citations [Source: filename.pdf, Page: X]
3. Contradicting Evidence: CT findings contradicting {disease}, features suggesting alternatives, exclusion criteria with citations
4. Diagnostic Workup: Additional imaging (MRI/PET-CT/contrast), lab tests, biopsy, follow-up protocols, timing
5. Safety & Clinical Guidance: Immediate actions, monitoring, red flags, when to seek urgent care, conservative management
6. Diagnostic Pathway: Step-by-step approach, follow-up timeline, multidisciplinary consultations, treatment vs. further workup decisions

Requirements:
- Cite EVERY claim: [Source: filename.pdf, Page: X]
- Use ONLY information from provided documents (no general knowledge)
- For each differential: condition name, supporting evidence, contradicting evidence, distinguishing features, diagnostic steps
- Emphasize safety and thorough evaluation given uncertainty
- If info missing, state: "Information not found in provided documents"
- Format: Markdown headers (##), structured per differential, bullet points, professional medical style"""

    try:
        rag_result = rag_chain.invoke({"input": query})
        rag_answer = rag_result.get("answer") or rag_result.get("output") or rag_result.get("result", "")
        source_docs = rag_result.get("context") or rag_result.get("source_documents") or []
    except Exception as e:
        rag_answer = f"RAG failure: {e}"
        source_docs = []

    # Step 3: Build final report with sources
    formatted_sources = [
        {
            "source": getattr(d, "metadata", {}).get("source", "Unknown"),
            "page": getattr(d, "metadata", {}).get("page", "Unknown"),
        } for d in source_docs
    ]
    
    final_report = build_report(
        disease=disease if is_confident else f"Uncertain ({disease})",
        confidence=confidence,
        rag_answer=rag_answer,
        sources=formatted_sources if formatted_sources else None
    )

    return {
        "prediction": {
            "class_id": class_id,
            "disease": disease,
            "confidence": confidence,
            "is_confident": is_confident,
            "confidence_threshold": confidence_threshold,
        },
        "rag_answer": rag_answer,
        "sources": [
            {
                "source": getattr(d, "metadata", {}).get("source"),
                "page": getattr(d, "metadata", {}).get("page"),
            } for d in source_docs
        ],
        "final_report": final_report,
    }

# Example usage

if __name__ == "__main__":
    from .config import DATA_DIR
    
    example_image = DATA_DIR / "10.jpg"
    if not example_image.exists():
        print(f"Example image not found: {example_image}")
        print("Please provide a valid image path.")
    else:
        result = run_pipeline_on_dicom(dicom_path=str(example_image))
        print(result["final_report"])
