"""
Prediction Engine for AI Fake Financial News Detection
Supports both TF-IDF and BERT-based inference
Loads trained model and provides classification with confidence scores
"""

import os
import sys
import json
import joblib
import numpy as np
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.preprocessor import TextPreprocessor


class PredictionResult:
    """Data class for prediction results"""
    
    def __init__(self, label: str, confidence: float, timestamp: str = None):
        self.label = label
        self.confidence = confidence
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def to_dict(self) -> dict:
        return {
            'label': self.label,
            'confidence': self.confidence,
            'timestamp': self.timestamp
        }
    
    def __str__(self):
        return f"[{self.timestamp}] {self.label} (Confidence: {self.confidence:.2%})"


class NewsClassifier:
    """
    Fake Financial News Classifier
    
    Automatically detects whether to use BERT or TF-IDF based on
    saved model metadata. Falls back gracefully.
    """
    
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(PROJECT_ROOT, 'models')
        
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None
        self.bert_model = None
        self.preprocessor = TextPreprocessor()
        self.feature_type = None
        self.is_loaded = False
        self.label_map = {0: 'FAKE', 1: 'REAL'}
    
    def load_model(self):
        """Load the trained model, vectorizer/BERT, and metadata"""
        model_path = os.path.join(self.model_dir, 'model.pkl')
        meta_path = os.path.join(self.model_dir, 'model_metadata.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}\n"
                f"Please run train_model.py first."
            )
        
        self.model = joblib.load(model_path)
        
        # Load metadata to determine feature type
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            self.feature_type = metadata.get('feature_type', 'TF-IDF')
            bert_model_name = metadata.get('bert_model_name')
        else:
            self.feature_type = 'TF-IDF'
            bert_model_name = None
        
        if self.feature_type == 'BERT' and bert_model_name:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"🤖 Loading BERT model: {bert_model_name}")
                self.bert_model = SentenceTransformer(bert_model_name)
                print(f"✅ BERT model loaded successfully")
            except ImportError:
                print("⚠️ sentence-transformers not available. Falling back to TF-IDF.")
                self.feature_type = 'TF-IDF'
        
        if self.feature_type == 'TF-IDF':
            vectorizer_path = os.path.join(self.model_dir, 'vectorizer.pkl')
            if os.path.exists(vectorizer_path):
                self.vectorizer = joblib.load(vectorizer_path)
            else:
                raise FileNotFoundError(
                    f"Vectorizer not found at: {vectorizer_path}\n"
                    f"Please run train_model.py first."
                )
        
        self.is_loaded = True
        print(f"✅ Classifier loaded (mode: {self.feature_type})")
        return self
    
    def _get_features(self, text: str):
        """Extract features using the appropriate method"""
        if self.feature_type == 'BERT' and self.bert_model:
            embedding = self.bert_model.encode([text], normalize_embeddings=True)
            return np.array(embedding)
        else:
            processed = self.preprocessor.preprocess(text)
            return self.vectorizer.transform([processed])
    
    def predict(self, text: str) -> PredictionResult:
        """Classify a financial news article as REAL or FAKE"""
        if not self.is_loaded:
            self.load_model()
        
        features = self._get_features(text)
        
        prediction = self.model.predict(features)[0]
        label = self.label_map[prediction]
        
        # Get confidence score
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
            confidence = max(probabilities)
        elif hasattr(self.model, 'decision_function'):
            decision = self.model.decision_function(features)[0]
            confidence = 1 / (1 + np.exp(-abs(decision)))  # sigmoid
        else:
            confidence = 0.85
        
        return PredictionResult(label=label, confidence=confidence)
    
    def predict_batch(self, texts: list) -> list:
        """Classify multiple articles"""
        return [self.predict(text) for text in texts]
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        meta_path = os.path.join(self.model_dir, 'model_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                return json.load(f)
        return {'feature_type': self.feature_type or 'Unknown'}


if __name__ == "__main__":
    classifier = NewsClassifier()
    
    test_articles = [
        "RBI keeps repo rate unchanged at 6.5% in latest monetary policy review. MPC voted 4-2 in favor of status quo.",
        "BREAKING: All bank accounts will be frozen tomorrow! Withdraw money now! Government to seize deposits!",
        "Sensex closes at record high driven by banking and IT stocks. FIIs net buyers worth Rs 3500 crore.",
        "Government to seize all fixed deposits above Rs 5 lakh secretly. Act before midnight tonight!",
        "SEBI introduces new regulations for IPO listing to enhance investor protection and transparency.",
        "Every Indian citizen to receive Rs 15 lakh in bank account. Click link and share Aadhaar to claim.",
    ]
    
    print("🔍 AI Fake Financial News Detection - Predictions\n")
    for article in test_articles:
        result = classifier.predict(article)
        emoji = "✅" if result.label == "REAL" else "🚨"
        print(f"{emoji} {result}")
        print(f"   → {article[:70]}...\n")
