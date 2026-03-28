# Proposed System

## AI-Based Fake Financial News Detection Web Application

---

### 1. System Overview

The proposed system is an **AI-Based Fake Financial News Detection Web Application** designed specifically for the **Indian financial information environment**. The system aims to use **textual analysis** to classify financial news articles and detect fake or misleading financial content using **Artificial Intelligence (AI), Natural Language Processing (NLP), Machine Learning (ML), and Deep Learning** techniques.

### 2. Core Objectives

| # | Objective | Description |
|---|---|---|
| 1 | **Accurate Detection** | Classify financial news articles as **REAL** or **FAKE** with high accuracy using trained ML/DL models. |
| 2 | **Domain Specificity** | Tailored specifically for the Indian financial ecosystem — understanding SEBI, RBI, NSE/BSE terminology and context. |
| 3 | **Real-Time Analysis** | Provide instant classification results, enabling users to verify news before making financial decisions. |
| 4 | **Confidence Scoring** | Output a **prediction result along with a confidence score** for transparency and informed decision-making. |
| 5 | **User Accessibility** | Deliver a **fast, user-friendly, scalable, and domain-focused** web interface accessible to all user levels. |

### 3. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE (Frontend)                     │
│                     Streamlit / Flask Web Application                │
│    ┌──────────────────────────────────────────────────────────┐      │
│    │  📰 Input: Paste Financial News Article / Headline       │      │
│    │  🔍 Action: Click "Analyze" Button                      │      │
│    │  📊 Output: Real/Fake Label + Confidence Score (%)      │      │
│    └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     BACKEND PROCESSING ENGINE                        │
│                                                                      │
│  ┌───────────────┐    ┌─────────────────┐    ┌────────────────────┐  │
│  │ Text          │───▶│ Feature         │───▶│ Classification     │  │
│  │ Preprocessing │    │ Extraction      │    │ Model              │  │
│  │               │    │                 │    │                    │  │
│  │ • Tokenization│    │ • TF-IDF       │    │ • Logistic         │  │
│  │ • Stopword    │    │   Vectorization │    │   Regression       │  │
│  │   Removal     │    │ • Word          │    │         OR         │  │
│  │ • Stemming/   │    │   Embeddings   │    │ • BERT-based       │  │
│  │   Lemmatization│   │ • N-gram       │    │   Transformer      │  │
│  │ • Lowercasing │    │   Features     │    │   Model            │  │
│  └───────────────┘    └─────────────────┘    └────────┬───────────┘  │
│                                                       │              │
│                                               ┌───────▼───────────┐  │
│                                               │ Prediction Output │  │
│                                               │ • Label: Real/Fake│  │
│                                               │ • Confidence: 87% │  │
│                                               └───────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      DATA & MODEL LAYER                              │
│                                                                      │
│  ┌─────────────────────┐         ┌─────────────────────────────┐    │
│  │ Training Dataset    │         │ Pre-trained Model            │    │
│  │ • Financial news    │         │ • Serialized ML model        │    │
│  │   articles          │         │   (.pkl / .pt / .h5)         │    │
│  │ • Labeled: Real/Fake│         │ • Ready for inference        │    │
│  └─────────────────────┘         └─────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

### 4. Key Features

#### 4.1 Intelligent Text Analysis
- The system utilizes **Natural Language Processing (NLP)** to preprocess and analyze text from financial news articles.
- Preprocessing includes **tokenization, stop-word removal, stemming/lemmatization**, and text normalization.

#### 4.2 Machine Learning Classification
- A trained classification model — either **Logistic Regression** or a **BERT-based deep learning model** — analyzes the input text.
- The model is **trained and evaluated on curated financial news datasets** to ensure domain relevance and accuracy.

#### 4.3 Prediction with Confidence Score
- The frontend web interface allows users to **paste financial news content** and instantly receive:
  - A **prediction result**: `REAL` or `FAKE`
  - A **confidence score** (e.g., 94.2% confident the article is fake)

#### 4.4 Indian Financial Context
- The system is specifically tuned to understand the nuances of **Indian financial reporting**, including references to:
  - Regulatory bodies: **SEBI, RBI, IRDAI**
  - Stock exchanges: **NSE, BSE**
  - Financial instruments: **Mutual Funds, IPOs, Government Securities**

#### 4.5 User-Friendly Web Interface
- Built using **Streamlit** or **Flask**, the interface is designed to be:
  - **Fast** — Results generated in seconds
  - **User-friendly** — Clean, intuitive design requiring no technical knowledge
  - **Scalable** — Can handle multiple concurrent users
  - **Domain-focused** — Tailored UI elements for financial news verification

### 5. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Programming Language** | Python 3.x | Core development language |
| **ML Libraries** | scikit-learn, pandas, numpy | Data processing, model training, evaluation |
| **NLP Libraries** | NLTK, spaCy | Text preprocessing, tokenization, lemmatization |
| **Deep Learning** | Transformers (Hugging Face), PyTorch / TensorFlow | BERT-based model training and inference |
| **Web Framework** | Streamlit or Flask / FastAPI | Frontend interface and API endpoints |
| **Model Serialization** | Pickle / Joblib / PyTorch Save | Saving and loading trained models |
| **Development Tools** | VS Code, Jupyter Notebook | Code development and experimentation |
| **Version Control** | Git & GitHub | Source code management |

### 6. Workflow Diagram

```
┌─────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  User   │────▶│  Web App     │────▶│  Preprocessing  │────▶│  Feature     │
│  Input  │     │  Interface   │     │  Pipeline       │     │  Extraction  │
└─────────┘     └──────────────┘     └─────────────────┘     └──────┬───────┘
                                                                     │
     ┌───────────────────────────────────────────────────────────────┘
     ▼
┌──────────────┐     ┌──────────────┐     ┌─────────────────────────┐
│  ML / DL     │────▶│  Prediction  │────▶│  Display Result         │
│  Model       │     │  Engine      │     │  • REAL / FAKE Label    │
│  Inference   │     │              │     │  • Confidence Score (%) │
└──────────────┘     └──────────────┘     └─────────────────────────┘
```

### 7. Expected Outcomes

1. **High Accuracy Classification** — Target accuracy of ≥ 90% on test datasets.
2. **Sub-second Response Time** — Real-time prediction for user queries.
3. **Transparent Results** — Confidence scores help users gauge reliability.
4. **Scalable Deployment** — Can be hosted on cloud platforms (Heroku, AWS, etc.).
5. **Research Contribution** — Demonstrates the application of NLP/ML in financial misinformation detection for the Indian context.

### 8. Advantages Over Existing System

| Aspect | Existing System | Proposed System |
|---|---|---|
| Detection Method | Manual / Rule-based | AI + NLP + ML/DL |
| Speed | Hours to days | Seconds |
| Domain Focus | Generic | Indian Financial Context |
| Confidence Score | Not available | Provided with every prediction |
| Scalability | Low | High (web-based, cloud-ready) |
| User Interface | None / CLI | Modern Web Application |
| Accuracy | Inconsistent | ≥ 90% target accuracy |

---

*Project: AI-Based Fake Financial News Detection System*
*Course: Bachelor of Computer Applications (BCA) — Final Year Project*
