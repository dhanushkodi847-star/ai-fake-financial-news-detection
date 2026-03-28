"""
Model Training Module for AI Fake Financial News Detection
Supports both TF-IDF and BERT-based sentence embeddings
Trains and evaluates multiple ML classification models
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.preprocessor import TextPreprocessor
from src.features.feature_extractor import FeatureExtractor


def load_dataset(data_path):
    """Load and validate the financial news dataset"""
    print("=" * 60)
    print("📂 LOADING DATASET")
    print("=" * 60)
    
    df = pd.read_csv(data_path)
    df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    
    print(f"   Total samples: {len(df)}")
    print(f"   Real news:     {len(df[df['label'] == 1])}")
    print(f"   Fake news:     {len(df[df['label'] == 0])}")
    
    return df


def preprocess_data(df):
    """Apply NLP preprocessing to all texts"""
    print("\n" + "=" * 60)
    print("🔧 PREPROCESSING TEXT DATA")
    print("=" * 60)
    
    preprocessor = TextPreprocessor()
    df['processed_text'] = preprocessor.preprocess_batch(df['full_text'].tolist())
    
    print(f"   ✅ Preprocessed {len(df)} articles")
    return df, preprocessor


def extract_tfidf_features(df):
    """Extract TF-IDF features"""
    print("\n" + "=" * 60)
    print("📊 EXTRACTING TF-IDF FEATURES")
    print("=" * 60)
    
    extractor = FeatureExtractor(max_features=5000, ngram_range=(1, 2))
    X = extractor.fit_transform(df['processed_text'].tolist())
    y = df['label'].values
    
    print(f"   Feature matrix shape: {X.shape}")
    return X, y, extractor


def extract_bert_features(df):
    """Extract BERT sentence embeddings using sentence-transformers"""
    print("\n" + "=" * 60)
    print("🤖 EXTRACTING BERT SENTENCE EMBEDDINGS")
    print("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = 'all-MiniLM-L6-v2'
        print(f"   Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        
        texts = df['full_text'].tolist()
        print(f"   Encoding {len(texts)} articles...")
        
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )
        
        X = np.array(embeddings)
        y = df['label'].values
        
        print(f"   ✅ Embedding shape: {X.shape}")
        print(f"   Embedding dimension: {X.shape[1]}")
        
        return X, y, model_name
    
    except ImportError:
        print("   ⚠️ sentence-transformers not installed. Falling back to TF-IDF.")
        return None, None, None


def train_and_evaluate(X, y, feature_type="TF-IDF"):
    """Train multiple models and compare performance"""
    print("\n" + "=" * 60)
    print(f"🤖 TRAINING & EVALUATING MODELS ({feature_type})")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set:     {X_test.shape[0]} samples")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, random_state=42),
    }
    
    # Add LinearSVC for BERT (works well with dense features)
    if feature_type == "BERT":
        models['Linear SVC'] = LinearSVC(max_iter=2000, random_state=42)
    else:
        models['Naive Bayes'] = MultinomialNB(alpha=0.5)
    
    best_model = None
    best_model_name = ""
    best_accuracy = 0
    results = {}
    
    for name, model in models.items():
        print(f"\n   📌 Training: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        results[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'cv_mean': round(cv_scores.mean(), 4),
            'cv_std': round(cv_scores.std(), 4)
        }
        
        print(f"      Accuracy:   {acc:.4f}")
        print(f"      Precision:  {prec:.4f}")
        print(f"      Recall:     {rec:.4f}")
        print(f"      F1 Score:   {f1:.4f}")
        print(f"      CV Mean:    {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name
    
    print("\n" + "=" * 60)
    print(f"🏆 BEST MODEL: {best_model_name} ({feature_type})")
    print("=" * 60)
    
    y_pred_best = best_model.predict(X_test)
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred_best, target_names=['FAKE', 'REAL']))
    
    return best_model, best_model_name, results


def save_artifacts(model, extractor, model_dir, feature_type, bert_model_name, results, best_name):
    """Save model, vectorizer, and metadata"""
    print("\n" + "=" * 60)
    print("💾 SAVING MODEL ARTIFACTS")
    print("=" * 60)
    
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'model.pkl')
    joblib.dump(model, model_path)
    print(f"   ✅ Model saved to: {model_path}")
    
    if feature_type == "TF-IDF" and extractor:
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        extractor.save(vectorizer_path)
    
    # Save metadata
    metadata = {
        'feature_type': feature_type,
        'bert_model_name': bert_model_name,
        'best_model_name': best_name,
        'results': results
    }
    meta_path = os.path.join(model_dir, 'model_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Metadata saved to: {meta_path}")


def main():
    """Main training pipeline with BERT + TF-IDF comparison"""
    print("\n" + "🔥" * 30)
    print("   AI FAKE FINANCIAL NEWS DETECTION - MODEL TRAINING")
    print("   Enhanced with BERT Sentence Embeddings")
    print("🔥" * 30 + "\n")
    
    data_path = os.path.join(PROJECT_ROOT, 'data', 'financial_news_dataset.csv')
    model_dir = os.path.join(PROJECT_ROOT, 'models')
    
    # Step 1: Load dataset
    df = load_dataset(data_path)
    
    # Step 2: Preprocess
    df, preprocessor = preprocess_data(df)
    
    # ── Try BERT first ──
    X_bert, y_bert, bert_model_name = extract_bert_features(df)
    
    if X_bert is not None:
        bert_model, bert_best_name, bert_results = train_and_evaluate(X_bert, y_bert, "BERT")
        bert_acc = bert_results[bert_best_name]['accuracy']
    else:
        bert_acc = 0
    
    # ── Also train TF-IDF ──
    X_tfidf, y_tfidf, tfidf_extractor = extract_tfidf_features(df)
    tfidf_model, tfidf_best_name, tfidf_results = train_and_evaluate(X_tfidf, y_tfidf, "TF-IDF")
    tfidf_acc = tfidf_results[tfidf_best_name]['accuracy']
    
    # ── Compare and save best ──
    print("\n" + "=" * 60)
    print("📊 FINAL COMPARISON")
    print("=" * 60)
    
    if X_bert is not None:
        print(f"   BERT Best:   {bert_best_name} → Accuracy: {bert_acc:.4f}")
    print(f"   TF-IDF Best: {tfidf_best_name} → Accuracy: {tfidf_acc:.4f}")
    
    if X_bert is not None and bert_acc >= tfidf_acc:
        print(f"\n   🏆 Winner: BERT ({bert_best_name})")
        save_artifacts(bert_model, None, model_dir, "BERT", bert_model_name, bert_results, bert_best_name)
    else:
        print(f"\n   🏆 Winner: TF-IDF ({tfidf_best_name})")
        save_artifacts(tfidf_model, tfidf_extractor, model_dir, "TF-IDF", None, tfidf_results, tfidf_best_name)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"   Models saved to: {model_dir}/")
    print(f"   Ready for inference!\n")


if __name__ == "__main__":
    main()
