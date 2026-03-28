# Software Requirements Specification (SRS)

## AI-Based Fake Financial News Detection System

---

### 1. Document Information

| Field | Details |
|---|---|
| **Project Title** | AI-Based Fake Financial News Detection System |
| **Course** | Bachelor of Computer Applications (BCA) — Final Year Project |
| **Version** | 1.0 |
| **Date** | March 2026 |

---

### 2. Software Components

#### 2.1 Programming Language & Core Libraries

| Component | Version | Purpose |
|---|---|---|
| **Python** | 3.9+ | Primary programming language for the entire application |
| **scikit-learn** | Latest | Machine learning model training, evaluation, and inference |
| **pandas** | Latest | Data manipulation, cleaning, and analysis |
| **numpy** | Latest | Numerical computations and array operations |

#### 2.2 Natural Language Processing (NLP)

| Component | Version | Purpose |
|---|---|---|
| **NLTK** | Latest | Tokenization, stopword removal, stemming, text processing |
| **spaCy** | Latest | Advanced NLP pipeline — lemmatization, named entity recognition |

#### 2.3 Deep Learning (Optional / Advanced)

| Component | Version | Purpose |
|---|---|---|
| **Transformers** (Hugging Face) | Latest | BERT-based pre-trained models for text classification |
| **PyTorch / TensorFlow** | Latest | Deep learning framework for model training and inference |

#### 2.4 Web Framework

| Component | Version | Purpose |
|---|---|---|
| **Streamlit** | Latest | Rapid prototyping of interactive web-based data applications |
| **Flask / FastAPI** | Latest | Lightweight web server for REST API endpoints and serving |

#### 2.5 Development & Debugging Tools

| Component | Purpose |
|---|---|
| **VS Code** | Primary Integrated Development Environment (IDE) |
| **Jupyter Notebook** | Interactive development, experimentation, and visualization |
| **PyCharm** | Alternative IDE for structured Python development |

#### 2.6 Browser & Testing

| Component | Purpose |
|---|---|
| **Chrome / Edge / Firefox** | Testing the web application interface across browsers |
| **Postman** (Optional) | API endpoint testing during Flask/FastAPI development |

#### 2.7 Version Control

| Component | Purpose |
|---|---|
| **Git** | Local version control and change tracking |
| **GitHub** | Remote repository hosting and collaboration |

---

### 3. Software Requirements

#### 3.1 Operating System

| OS | Support Status |
|---|---|
| **Windows 10/11** | ✅ Fully Supported (Primary Development Platform) |
| **Linux** (Ubuntu 20.04+) | ✅ Fully Supported |
| **macOS** (Monterey+) | ✅ Fully Supported |

#### 3.2 Runtime Requirements

| Requirement | Specification |
|---|---|
| **Python Version** | 3.9 or higher |
| **pip** | Latest version (package manager) |
| **Virtual Environment** | `venv` or `conda` recommended |
| **Internet Connection** | Required for initial setup, model download, and optional API calls |

---

### 4. Hardware Requirements

#### 4.1 Minimum Requirements

| Component | Specification |
|---|---|
| **Processor** | Intel Core i5 (8th Gen) or equivalent |
| **RAM** | 8 GB |
| **Storage** | 10 GB free disk space |
| **GPU** | Not required (CPU inference supported) |

#### 4.2 Recommended Requirements (for BERT/Deep Learning)

| Component | Specification |
|---|---|
| **Processor** | Intel Core i7 (10th Gen) / AMD Ryzen 7 or higher |
| **RAM** | 16 GB |
| **Storage** | 20 GB free disk space (SSD preferred) |
| **GPU** | NVIDIA GPU with CUDA support (e.g., GTX 1650 or higher) |

---

### 5. Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-01 | User shall be able to input/paste a financial news article or headline | **High** |
| FR-02 | System shall preprocess the input text (tokenization, stop-word removal, etc.) | **High** |
| FR-03 | System shall classify the input as REAL or FAKE using the trained ML model | **High** |
| FR-04 | System shall display a confidence score (%) along with the classification | **High** |
| FR-05 | System shall provide a clean, responsive web interface | **Medium** |
| FR-06 | System shall handle invalid or empty inputs gracefully with error messages | **Medium** |
| FR-07 | System shall support batch analysis of multiple articles (optional) | **Low** |

### 6. Non-Functional Requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-01 | **Performance**: Prediction response time | < 3 seconds |
| NFR-02 | **Accuracy**: Model classification accuracy on test data | ≥ 90% |
| NFR-03 | **Usability**: Intuitive interface requiring no technical expertise | Pass |
| NFR-04 | **Scalability**: Support for concurrent users | ≥ 10 simultaneous users |
| NFR-05 | **Portability**: Cross-platform compatibility | Windows, Linux, macOS |
| NFR-06 | **Reliability**: System uptime during demonstration | ≥ 99% |

---

### 7. Dependencies Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
│              Streamlit / Flask Web Application               │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   ML / NLP ENGINE                            │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌────────────┐  │
│  │scikit-  │  │ NLTK /   │  │Transformers│  │ PyTorch /  │  │
│  │ learn   │  │ spaCy    │  │(Hugging   │  │TensorFlow  │  │
│  │         │  │          │  │  Face)    │  │            │  │
│  └─────────┘  └──────────┘  └───────────┘  └────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    DATA LAYER                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐  │
│  │  pandas   │  │  numpy   │  │  Dataset (CSV / JSON)    │  │
│  └──────────┘  └──────────┘  └──────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   PLATFORM LAYER                             │
│            Python 3.9+ | pip | venv / conda                  │
│          Windows / Linux / macOS                             │
└─────────────────────────────────────────────────────────────┘
```

---

*Project: AI-Based Fake Financial News Detection System*
*Course: Bachelor of Computer Applications (BCA) — Final Year Project*
