"""
Text Preprocessing Module for AI Fake Financial News Detection
Handles text cleaning, tokenization, stopword removal, and lemmatization
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


class TextPreprocessor:
    """
    NLP Text Preprocessing Pipeline for Financial News
    
    Pipeline Steps:
    1. Lowercasing
    2. URL removal
    3. Special character & number removal
    4. Tokenization
    5. Stopword removal
    6. Lemmatization
    7. Rejoining tokens
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Keep financial domain words that might be in stopwords
        self.financial_keep_words = {
            'not', 'no', 'nor', 'against', 'above', 'below', 'up', 'down',
            'over', 'under', 'few', 'more', 'most', 'all', 'any', 'both'
        }
        self.stop_words -= self.financial_keep_words
    
    def clean_text(self, text: str) -> str:
        """Remove URLs, special characters, numbers, and lowercase the text"""
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove numbers (but keep words with numbers like '5G')
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove special characters and punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> list:
        """Tokenize text into words"""
        try:
            return word_tokenize(text)
        except Exception:
            return text.split()
    
    def remove_stopwords(self, tokens: list) -> list:
        """Remove English stopwords while keeping financial domain words"""
        return [token for token in tokens if token not in self.stop_words and len(token) > 1]
    
    def lemmatize(self, tokens: list) -> list:
        """Lemmatize tokens to their base form"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline:
        clean → tokenize → remove stopwords → lemmatize → rejoin
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text string
        """
        # Step 1: Clean text
        cleaned = self.clean_text(text)
        
        # Step 2: Tokenize
        tokens = self.tokenize(cleaned)
        
        # Step 3: Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Step 4: Lemmatize
        tokens = self.lemmatize(tokens)
        
        # Step 5: Rejoin
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: list) -> list:
        """Preprocess a batch of texts"""
        return [self.preprocess(text) for text in texts]


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    sample_text = """BREAKING: The Reserve Bank of India (RBI) has announced 
    a complete ban on all bank transactions for 30 days! 
    Visit https://fake-news.com for more details. ALL ATMs will be SHUT DOWN!!!"""
    
    print("Original:", sample_text)
    print("\nCleaned:", preprocessor.clean_text(sample_text))
    print("\nFull Pipeline:", preprocessor.preprocess(sample_text))
