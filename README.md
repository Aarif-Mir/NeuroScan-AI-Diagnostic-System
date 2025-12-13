# 🏥 CT RAG Agent - AI-Powered CT Scan Diagnosis System

AI system that combines CNN models with RAG to provide evidence-based diagnostic reports for CT brain scans. Classifies brain conditions (Aneurysm, Cancer, Tumor) and generates medical reports using peer-reviewed literature.

## Features

- **Image Classification**: ResNet50 CNN for 3-class classification (Aneurysm, Cancer, Tumor)
- **RAG Reports**: Evidence-based medical reports with source citations
- **Multi-format Support**: DICOM (.dcm), JPG, PNG
- **Web Interface**: Streamlit-based UI with chat assistant
- **Anti-Hallucination**: Strict context-only responses with citations

## Installation

### Prerequisites
- Python 3.12+
- Groq API Key

### Setup

```bash
# Clone repository
git clone https://github.com/Aarif-Mir/NeuroScan-AI-Diagnostic-System
cd NeuroScan-AI-Diagnostic-System

# Install dependencies (using UV)
pip install uv
uv sync

# Or using pip
pip install -e .
```

### Environment Variables

Create `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Optional variables (defaults provided):
- `CONFIDENCE_THRESHOLD=0.5`
- `MODEL_PATH=models/tf_cnn/resnet_model3.keras`
- `CHROMA_DIR=chroma_db`
- `PDF_DIR=data/rag_pdfs`
- `LLM_MODEL_NAME=llama-3.1-8b-instant`
```

## Usage

### Web Interface

```bash
streamlit run frontend1.py
```

Access at `http://localhost:8501`

**Modes:**
1. **Image Analysis**: Upload CT scan → Get diagnosis + medical report
2. **Chat Assistant**: Ask follow-up questions about diagnosis

## Project Structure

```
NeuroScan-AI-Diagnostic-System/
├── src/
│   ├── config.py              # Configuration
│   ├── preprocessing.py       # Image preprocessing
│   ├── inference.py           # CNN inference
│   ├── rag_ingest.py          # PDF ingestion
│   ├── retriever.py           # RAG chain
│   └── diagnosis_pipeline.py  # Main pipeline
├── frontend.py              # Streamlit UI
├── models/tf_cnn/             # Trained models
├── data/rag_pdfs/            # Medical literature PDFs
└── chroma_db/                # Vector database (auto-generated)
```

## Configuration

Key settings in `src/config.py` (override via environment variables):

- `CONFIDENCE_THRESHOLD`: Minimum confidence for diagnosis (default: 0.5)
- `MODEL_PATH`: CNN model path
- `CHROMA_DIR`: Vector database directory
- `PDF_DIR`: Medical PDF storage
- `RAG_K_RETRIEVE`: Number of documents to retrieve (default: 20)

## Model Details

Three CNN architectures were trained and evaluated:

### ResNet50 (Active Model)
- **Base**: ResNet50 (ImageNet pre-trained)
- **Input**: 224x224x3 RGB images
- **Custom Layers**: Dense(124) → Dense(64) → Dense(16) → Dense(3, softmax)
- **Output**: 3-class classification (Aneurysm, Cancer, Tumor)
- **Performance**: ~99.08% training accuracy
- **Optimizer**: Adam (lr=0.001)

### VGG16
- **Base**: VGG16 (ImageNet pre-trained)
- **Input**: 256x256x3 RGB images
- **Custom Layers**: Dense(124) → Dense(64) → Dense(16) → Dense(3, softmax)
- **Output**: 3-class classification
- **Optimizer**: SGD (lr=0.001)

### EfficientNetB7
- **Base**: EfficientNetB7 (ImageNet pre-trained)
- **Input**: 224x224x3 RGB images
- **Custom Layers**: Dense(124) → Dense(64) → Dense(16) → Dense(3, softmax)
- **Output**: 3-class classification
- **Optimizer**: Adam (lr=0.001)

**Note**: ResNet50 is the default active model. All models use transfer learning with frozen base layers and custom classification heads.

## ⚠️ Disclaimer

**FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY**

- NOT a substitute for professional medical diagnosis
- NOT approved for clinical use
- Always consult qualified radiologists and physicians
- Use at your own risk

---

**Built for medical AI research**
