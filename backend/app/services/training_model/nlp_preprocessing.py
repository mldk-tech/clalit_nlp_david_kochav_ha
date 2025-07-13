import pandas as pd
import spacy
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK resources if not already present
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

class NLPPreprocessor:
    """
    NLP preprocessing pipeline for clinical text using spaCy and NLTK.
    """
    def __init__(self, language_model: str = 'en_core_web_sm'):
        self.spacy_nlp = spacy.load(language_model, disable=["ner", "parser"])
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return nltk.word_tokenize(text)

    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def stem(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(token) for token in tokens]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stop_words]

    def spacy_lemmatize(self, text: str) -> List[str]:
        doc = self.spacy_nlp(text)
        return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    def preprocess_text(self, text: str, use_spacy: bool = True) -> Dict[str, Any]:
        """
        Full preprocessing: tokenization, lemmatization, stemming, stop word removal.
        Returns a dictionary with all representations.
        """
        if not text:
            return {
                'tokens': [],
                'lemmas': [],
                'stems': [],
                'no_stopwords': [],
                'spacy_lemmas': []
            }
        tokens = self.tokenize(text)
        lemmas = self.lemmatize(tokens)
        stems = self.stem(tokens)
        no_stopwords = self.remove_stopwords(tokens)
        spacy_lemmas = self.spacy_lemmatize(text) if use_spacy else []
        return {
            'tokens': tokens,
            'lemmas': lemmas,
            'stems': stems,
            'no_stopwords': no_stopwords,
            'spacy_lemmas': spacy_lemmas
        }

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'cleaned_summary') -> pd.DataFrame:
        """
        Apply NLP preprocessing to a DataFrame column.
        Adds new columns for tokens, lemmas, stems, and stopword removal.
        """
        logger.info(f"Applying NLP preprocessing to column: {text_column}")
        nlp_results = df[text_column].apply(self.preprocess_text)
        df['tokens'] = nlp_results.apply(lambda x: x['tokens'])
        df['lemmas'] = nlp_results.apply(lambda x: x['lemmas'])
        df['stems'] = nlp_results.apply(lambda x: x['stems'])
        df['no_stopwords'] = nlp_results.apply(lambda x: x['no_stopwords'])
        df['spacy_lemmas'] = nlp_results.apply(lambda x: x['spacy_lemmas'])
        logger.info("NLP preprocessing complete.")
        return df

def main():
    from data_loader import AppointmentDataLoader
    from data_preprocessing import DataPreprocessor
    # Load and preprocess data
    loader = AppointmentDataLoader()
    df = loader.to_dataframe()
    preprocessor = DataPreprocessor(df)
    processed_df = preprocessor.preprocess_data()
    # Apply NLP preprocessing
    nlp = NLPPreprocessor()
    nlp_df = nlp.preprocess_dataframe(processed_df)
    print(nlp_df[['cleaned_summary', 'tokens', 'lemmas', 'stems', 'no_stopwords', 'spacy_lemmas']].head())
    # Save to CSV
    nlp_df[['cleaned_summary', 'tokens', 'lemmas', 'stems', 'no_stopwords', 'spacy_lemmas']].to_csv('nlp_cleaned_data.csv', index=False)
if __name__ == "__main__":
    main() 