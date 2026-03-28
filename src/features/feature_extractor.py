"""
Feature Extraction Module for AI Fake Financial News Detection
Uses TF-IDF Vectorization to convert text into numerical features
"""

import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:
    """
    TF-IDF based Feature Extractor for text classification
    
    Converts preprocessed text into numerical feature vectors
    using Term Frequency-Inverse Document Frequency (TF-IDF)
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize the TF-IDF Vectorizer
        
        Args:
            max_features: Maximum number of features (vocabulary size)
            ngram_range: Range of n-grams (1,2) = unigrams + bigrams
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        self.is_fitted = False
    
    def fit(self, texts):
        """Fit the vectorizer on training texts"""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts):
        """Transform texts into TF-IDF feature vectors"""
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted before transforming. Call fit() first.")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """Fit and transform in a single step"""
        result = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return result
    
    def get_feature_names(self):
        """Get the vocabulary (feature names)"""
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted first.")
        return self.vectorizer.get_feature_names_out()
    
    def save(self, filepath):
        """Save the fitted vectorizer to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)
        print(f"✅ Vectorizer saved to: {filepath}")
    
    def load(self, filepath):
        """Load a fitted vectorizer from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vectorizer file not found: {filepath}")
        self.vectorizer = joblib.load(filepath)
        self.is_fitted = True
        print(f"✅ Vectorizer loaded from: {filepath}")
        return self


if __name__ == "__main__":
    # Test the feature extractor
    extractor = FeatureExtractor(max_features=100)
    
    sample_texts = [
        "rbi keeps repo rate unchanged percent",
        "sebi introduces new regulations mutual fund",
        "breaking rbi ban all bank transactions",
        "stock market crash percent tomorrow analyst",
    ]
    
    features = extractor.fit_transform(sample_texts)
    print(f"Feature matrix shape: {features.shape}")
    print(f"Number of features: {len(extractor.get_feature_names())}")
    print(f"Top features: {list(extractor.get_feature_names()[:10])}")
