# 🔍 AI-Based Fake Financial News Detection System

<div align="center">

**A Machine Learning-Powered Web Application for Detecting Fake Financial News**

*Bachelor of Computer Applications (BCA) — Final Year Project 2025-2026*

---

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![BERT](https://img.shields.io/badge/BERT-Sentence_Embeddings-blueviolet?style=for-the-badge&logo=huggingface&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red?style=for-the-badge&logo=streamlit&logoColor=white)

</div>

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [License](#license)

---

## 📖 About the Project

In the digital age, **fake financial news** poses a critical threat to market integrity, investor confidence, and economic stability. This project presents an **AI-powered web application** specifically designed for the Indian financial ecosystem that can classify financial news articles as **REAL** or **FAKE** using advanced NLP and Machine Learning techniques.

The system leverages **BERT Sentence Embeddings** (`all-MiniLM-L6-v2`) for feature extraction and **Random Forest** classifier for prediction, providing users with:
- A prediction label (**REAL** or **FAKE**)
- A **confidence score** for transparency

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **AI-Powered Detection** | Uses trained ML/DL models for accurate classification |
| 📊 **Confidence Score** | Provides probability-based confidence with every prediction |
| ⚡ **Real-Time Analysis** | Sub-3-second response times for instant verification |
| 🇮🇳 **Indian Financial Focus** | Tailored for SEBI, RBI, NSE/BSE, and Indian market context |
| 🌐 **Web Interface** | Clean, intuitive Streamlit/Flask-based interface |
| 📝 **NLP Pipeline** | Comprehensive text preprocessing (tokenization, lemmatization, etc.) |

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Language** | Python 3.9+ |
| **Embeddings** | BERT (`all-MiniLM-L6-v2`) via sentence-transformers |
| **ML** | scikit-learn (Random Forest, Logistic Regression, Gradient Boosting, Linear SVC) |
| **NLP** | NLTK (tokenization, lemmatization, stopword removal) |
| **Deep Learning** | PyTorch (BERT backend) |
| **Data** | pandas, numpy |
| **Web Framework** | Streamlit |
| **IDE** | VS Code |
| **Version Control** | Git & GitHub |

---

## 📁 Project Structure

```
AI-Fake-Financial/
│
├── 📄 README.md                          # Project overview (this file)
├── 📄 requirements.txt                   # Python dependencies
│
├── 📂 docs/                              # Project documentation
│   ├── 01_Problem_Statement.md           # Problem definition & motivation
│   ├── 02_Existing_System.md             # Analysis of current approaches
│   ├── 03_Proposed_System.md             # Proposed solution & architecture
│   ├── 04_Software_Requirements_Specification.md  # SRS document
│   ├── 05_Project_Synopsis.md            # Project abstract & synopsis
│   └── 06_System_Architecture_and_Design.md       # UML & system diagrams
│
├── 📂 data/                              # Datasets
│   └── financial_news_dataset.csv        # 550+ Indian financial news articles
│
├── 📂 models/                            # Trained ML models
│   ├── model.pkl                         # Best trained model (Random Forest + BERT)
│   ├── model_metadata.json               # Model metadata & evaluation results
│   └── vectorizer.pkl                    # TF-IDF vectorizer (fallback)
│
├── 📂 src/                               # Source code
│   ├── preprocessing/                    # NLP preprocessing modules
│   │   └── preprocessor.py              # Text cleaning, tokenization, lemmatization
│   ├── features/                         # Feature extraction modules
│   │   └── feature_extractor.py         # TF-IDF feature extractor
│   ├── models/                           # Model training & prediction
│   │   ├── train_model.py               # Training pipeline (BERT + TF-IDF comparison)
│   │   └── predictor.py                 # Inference engine with BERT/TF-IDF support
│   ├── data/                             # Dataset generation
│   │   └── generate_dataset.py          # Script to generate training data
│   └── app/                              # Web application
│       └── app.py                        # Streamlit web app (premium dark UI)
│
└── 📂 .venv/                             # Virtual environment
```

---

## 📚 Documentation

| Document | Description | Link |
|---|---|---|
| **Problem Statement** | Problem definition and motivation | [View](docs/01_Problem_Statement.md) |
| **Existing System** | Analysis of current approaches and limitations | [View](docs/02_Existing_System.md) |
| **Proposed System** | Detailed proposed solution with architecture | [View](docs/03_Proposed_System.md) |
| **Software Requirements** | Complete SRS with HW/SW requirements | [View](docs/04_Software_Requirements_Specification.md) |
| **Project Synopsis** | Abstract, objectives, methodology, modules | [View](docs/05_Project_Synopsis.md) |
| **System Architecture** | DFDs, Use Cases, Class & Sequence diagrams | [View](docs/06_System_Architecture_and_Design.md) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ai-fake-financial-news-detection.git
cd ai-fake-financial-news-detection

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"

# 5. Train the model
python src/models/train_model.py

# 6. Run the application
python -m streamlit run src/app/app.py
```

> **Note:** Step 5 trains both BERT and TF-IDF models, compares them, and saves the best one automatically. The BERT model (`all-MiniLM-L6-v2`) will be downloaded on first run (~80 MB).

---

## 💻 Usage

1. **Open** the web application in your browser
2. **Paste** a financial news article or headline into the input field
3. **Click** the "Analyze" button
4. **View** the classification result (REAL / FAKE) along with the confidence score

---

## 📄 License

This project is developed as an academic project for the **Bachelor of Computer Applications (BCA)** program.

---

<div align="center">

**Made with ❤️ for BCA Final Year Project**

*AI-Based Fake Financial News Detection System | 2025-2026*

</div>
