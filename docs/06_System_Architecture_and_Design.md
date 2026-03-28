# System Architecture & Design

## AI-Based Fake Financial News Detection System

---

### 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                    🌐 PRESENTATION LAYER                            │
│                                                                     │
│    ┌─────────────────────────────────────────────────────────┐      │
│    │          Streamlit / Flask Web Application               │      │
│    │                                                         │      │
│    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │      │
│    │  │  Input Form   │  │  Result Card  │  │  History     │  │      │
│    │  │  Component    │  │  Component    │  │  Component   │  │      │
│    │  └──────────────┘  └──────────────┘  └──────────────┘  │      │
│    └─────────────────────────────────────────────────────────┘      │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                    ⚙️ BUSINESS LOGIC LAYER                          │
│                                                                     │
│    ┌──────────────────────────────────────────────────────────┐     │
│    │                 API / Request Handler                     │     │
│    └──────────────────────┬───────────────────────────────────┘     │
│                           │                                         │
│    ┌──────────────────────▼───────────────────────────────────┐     │
│    │              NLP Preprocessing Pipeline                   │     │
│    │  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │     │
│    │  │Tokenize │→│Stopwords │→│Lemmatize │→│Normalize   │  │     │
│    │  └─────────┘ └──────────┘ └──────────┘ └────────────┘  │     │
│    └──────────────────────┬───────────────────────────────────┘     │
│                           │                                         │
│    ┌──────────────────────▼───────────────────────────────────┐     │
│    │              Feature Extraction Engine                    │     │
│    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │     │
│    │  │  TF-IDF      │  │  Word        │  │  N-gram      │  │     │
│    │  │  Vectorizer  │  │  Embeddings  │  │  Features    │  │     │
│    │  └──────────────┘  └──────────────┘  └──────────────┘  │     │
│    └──────────────────────┬───────────────────────────────────┘     │
│                           │                                         │
│    ┌──────────────────────▼───────────────────────────────────┐     │
│    │              Classification Engine                       │     │
│    │  ┌────────────────────┐  ┌────────────────────────────┐ │     │
│    │  │ Logistic Regression│  │ BERT Transformer (Optional)│ │     │
│    │  └────────────────────┘  └────────────────────────────┘ │     │
│    └──────────────────────┬───────────────────────────────────┘     │
│                           │                                         │
│    ┌──────────────────────▼───────────────────────────────────┐     │
│    │              Prediction Output                           │     │
│    │         Label: REAL / FAKE  |  Confidence: 94.2%         │     │
│    └──────────────────────────────────────────────────────────┘     │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                    💾 DATA & MODEL LAYER                            │
│                                                                     │
│    ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│    │  Training Data   │  │  Trained      │  │  TF-IDF          │    │
│    │  (CSV / JSON)    │  │  Model (.pkl) │  │  Vectorizer(.pkl)│    │
│    └──────────────────┘  └──────────────┘  └──────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 2. Data Flow Diagram (Level 0 — Context Diagram)

```
                            ┌─────────────────────────┐
                            │                         │
   Financial News Article   │   AI Fake Financial     │   Classification Result
   ────────────────────────▶│   News Detection        │──────────────────────▶
      (User Input)          │   System                │    (REAL / FAKE +
                            │                         │     Confidence %)
                            └─────────────────────────┘
```

---

### 3. Data Flow Diagram (Level 1)

```
┌──────┐    News Text    ┌──────────────┐   Clean Text    ┌──────────────┐
│      │ ──────────────▶ │  1.0         │ ──────────────▶ │  2.0         │
│ User │                 │  Preprocess  │                  │  Extract     │
│      │                 │  Text        │                  │  Features    │
└──────┘                 └──────────────┘                  └──────┬───────┘
   ▲                                                              │
   │                                                              │
   │                                                     Feature Vector
   │                                                              │
   │     Result + Score  ┌──────────────┐   Prediction   ┌───────▼───────┐
   │ ◀────────────────── │  4.0         │ ◀───────────── │  3.0          │
   │                     │  Display     │                │  Classify     │
   │                     │  Results     │                │  (ML Model)   │
   │                     └──────────────┘                └───────────────┘
   │                           │
   │                           │
   └───────────────────────────┘
```

---

### 4. Use Case Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM BOUNDARY                               │
│                                                                  │
│                                                                  │
│  ┌─────────────┐         ┌──────────────────────────────┐       │
│  │             │         │  UC-01: Input News Article    │       │
│  │             │────────▶│                               │       │
│  │             │         └──────────────────────────────┘       │
│  │             │                                                 │
│  │             │         ┌──────────────────────────────┐       │
│  │    USER     │────────▶│  UC-02: View Classification  │       │
│  │  (Actor)    │         │         Result               │       │
│  │             │         └──────────────────────────────┘       │
│  │             │                                                 │
│  │             │         ┌──────────────────────────────┐       │
│  │             │────────▶│  UC-03: View Confidence      │       │
│  │             │         │         Score                 │       │
│  └─────────────┘         └──────────────────────────────┘       │
│                                                                  │
│                          ┌──────────────────────────────┐       │
│                          │  UC-04: Handle Invalid Input  │       │
│                          │         (Error Handling)      │       │
│                          └──────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### 5. Class Diagram

```
┌────────────────────────────┐       ┌────────────────────────────┐
│      TextPreprocessor      │       │     FeatureExtractor       │
├────────────────────────────┤       ├────────────────────────────┤
│ - tokenizer: NLTK          │       │ - vectorizer: TfidfVect.  │
│ - stopwords: set           │       │ - max_features: int       │
│ - lemmatizer: WordNetLemm. │       ├────────────────────────────┤
├────────────────────────────┤       │ + fit(corpus): void        │
│ + clean(text): str         │──────▶│ + transform(text): array  │
│ + tokenize(text): list     │       │ + fit_transform(): array  │
│ + remove_stopwords(): list │       └──────────┬─────────────────┘
│ + lemmatize(tokens): str   │                  │
└────────────────────────────┘                  │
                                                │
                                                ▼
┌────────────────────────────┐       ┌────────────────────────────┐
│      NewsClassifier        │       │      PredictionResult      │
├────────────────────────────┤       ├────────────────────────────┤
│ - model: sklearn.Model     │       │ - label: str (REAL/FAKE)  │
│ - model_path: str          │       │ - confidence: float       │
│ - is_loaded: bool          │       │ - timestamp: datetime     │
├────────────────────────────┤       ├────────────────────────────┤
│ + load_model(): void      │──────▶│ + to_dict(): dict         │
│ + predict(features): Result│       │ + __str__(): str          │
│ + get_confidence(): float  │       └────────────────────────────┘
│ + train(X, y): void        │
└────────────────────────────┘

┌────────────────────────────┐
│      WebApplication        │
├────────────────────────────┤
│ - app: Streamlit/Flask     │
│ - preprocessor: TextPrep.  │
│ - extractor: FeatureExtr.  │
│ - classifier: NewsClassif. │
├────────────────────────────┤
│ + run(): void              │
│ + handle_input(text): void │
│ + display_result(): void   │
│ + render_ui(): void        │
└────────────────────────────┘
```

---

### 6. Sequence Diagram

```
User              WebApp           Preprocessor       Extractor          Classifier
 │                  │                  │                  │                  │
 │  Input Article   │                  │                  │                  │
 │─────────────────▶│                  │                  │                  │
 │                  │  clean(text)     │                  │                  │
 │                  │─────────────────▶│                  │                  │
 │                  │                  │                  │                  │
 │                  │  cleaned_text    │                  │                  │
 │                  │◀─────────────────│                  │                  │
 │                  │                  │                  │                  │
 │                  │  transform(text)                    │                  │
 │                  │────────────────────────────────────▶│                  │
 │                  │                                     │                  │
 │                  │  feature_vector                     │                  │
 │                  │◀────────────────────────────────────│                  │
 │                  │                                                       │
 │                  │  predict(features)                                    │
 │                  │─────────────────────────────────────────────────────▶│
 │                  │                                                       │
 │                  │  PredictionResult (label + confidence)                │
 │                  │◀─────────────────────────────────────────────────────│
 │                  │                  │                  │                  │
 │  Display Result  │                  │                  │                  │
 │◀─────────────────│                  │                  │                  │
 │ (REAL/FAKE + %)  │                  │                  │                  │
 │                  │                  │                  │                  │
```

---

### 7. ER Diagram (Data Model)

```
┌─────────────────────────┐         ┌─────────────────────────┐
│     NewsArticle          │         │     PredictionLog        │
├─────────────────────────┤         ├─────────────────────────┤
│ PK  article_id          │         │ PK  log_id               │
│     title               │         │ FK  article_id           │
│     content             │────────▶│     predicted_label      │
│     source              │         │     confidence_score     │
│     published_date      │         │     model_version        │
│     actual_label        │         │     prediction_timestamp │
│     category            │         └─────────────────────────┘
└─────────────────────────┘

┌─────────────────────────┐
│     TrainedModel         │
├─────────────────────────┤
│ PK  model_id             │
│     model_name           │
│     model_type           │
│     accuracy             │
│     precision            │
│     recall               │
│     f1_score             │
│     training_date        │
│     file_path            │
└─────────────────────────┘
```

---

*Project: AI-Based Fake Financial News Detection System*
*Course: Bachelor of Computer Applications (BCA) — Final Year Project*
