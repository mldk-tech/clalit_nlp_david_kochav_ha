import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from gensim.models import Word2Vec
from typing import List, Dict, Any, Tuple, Optional
import logging
import scipy.sparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextFeatureEngineer:
    """
    Advanced text feature engineering: TF-IDF, embeddings, feature selection, importance.
    """
    def __init__(self, df: pd.DataFrame, text_column: str = 'cleaned_summary', label_column: str = 'outcome_encoded'):
        self.df = df.copy()
        self.text_column = text_column
        self.label_column = label_column
        self.tfidf_vectorizer = None
        self.tfidf_features = None
        self.embedding_model = None
        self.embedding_features = None
        self.selected_features = None
        self.feature_names = None

    def compute_tfidf(self, max_features: int = 100) -> pd.DataFrame:
        logger.info("Computing TF-IDF features...")
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df[self.text_column].fillna('').astype(str))
        # Use .toarray() only if tfidf_matrix is a scipy sparse matrix
        if scipy.sparse.issparse(tfidf_matrix):
            tfidf_array = tfidf_matrix.toarray()  # type: ignore[attr-defined]
        else:
            tfidf_array = np.array(tfidf_matrix)
        tfidf_columns = pd.Index([f"tfidf_{f}" for f in self.tfidf_vectorizer.get_feature_names_out()])
        tfidf_df = pd.DataFrame(tfidf_array, columns=tfidf_columns)
        self.tfidf_features = tfidf_df
        logger.info(f"TF-IDF features shape: {tfidf_df.shape}")
        return tfidf_df

    def compute_word2vec(self, vector_size: int = 50, window: int = 5, min_count: int = 1) -> pd.DataFrame:
        logger.info("Training Word2Vec model and computing embedding features...")
        # Tokenize summaries
        sentences = self.df[self.text_column].fillna('').apply(lambda x: x.split())
        self.embedding_model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=1, epochs=20)
        # Average word vectors for each summary
        def avg_vector(tokens):
            if self.embedding_model is not None:
                vectors = [self.embedding_model.wv[w] for w in tokens if w in self.embedding_model.wv]
            else:
                vectors = []
            if vectors:
                return np.mean(vectors, axis=0)
            else:
                return np.zeros(vector_size)
        embedding_matrix = np.vstack(list(sentences.apply(avg_vector).values))
        embedding_columns = pd.Index([f'w2v_{i}' for i in range(vector_size)])
        embedding_df = pd.DataFrame(embedding_matrix, columns=embedding_columns)
        self.embedding_features = embedding_df
        logger.info(f"Word2Vec embedding features shape: {embedding_df.shape}")
        return embedding_df

    def combine_features(self, additional_features: Optional[List[pd.DataFrame]] = None) -> pd.DataFrame:
        logger.info("Combining all features (TF-IDF, embeddings, clinical)...")
        features = [self.tfidf_features, self.embedding_features]
        if additional_features:
            features.extend(additional_features)
        # Filter out None values
        features = [f for f in features if f is not None]
        combined = pd.concat(features, axis=1)
        self.feature_names = combined.columns.tolist()
        logger.info(f"Combined feature shape: {combined.shape}")
        return combined

    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 30) -> pd.DataFrame:
        logger.info(f"Selecting top {k} features using univariate chi2...")
        # Only keep numeric columns
        X_numeric = X.select_dtypes(include=[np.number])
        selector = SelectKBest(chi2, k=min(k, X_numeric.shape[1]))
        X_selected = selector.fit_transform(np.abs(X_numeric), y)
        support = selector.get_support()
        if support is None:
            selected_cols = []
        else:
            selected_cols = [str(col) for col, keep in zip(X_numeric.columns, support) if keep]
        self.selected_features = selected_cols
        logger.info(f"Selected features: {selected_cols}")
        return pd.DataFrame(X_selected, columns=pd.Index(selected_cols))

    def feature_importance(self, X: pd.DataFrame, y: pd.Series, n_top: int = 20) -> List[Tuple[str, float]]:
        logger.info("Computing feature importance using RandomForest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:n_top]
        top_features = [(str(X.columns[i]), float(importances[i])) for i in indices]
        logger.info(f"Top {n_top} features: {top_features}")
        return top_features

    def permutation_importance(self, X: pd.DataFrame, y: pd.Series, n_top: int = 20) -> List[Tuple[str, float]]:
        logger.info("Computing permutation importance...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        result = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=1)
        importances_mean = getattr(result, 'importances_mean', None)
        if importances_mean is None:
            return []
        sorted_idx = importances_mean.argsort()[::-1][:n_top]
        top_perm = [(str(X.columns[i]), float(importances_mean[i])) for i in sorted_idx]
        logger.info(f"Top {n_top} permutation importances: {top_perm}")
        return top_perm

    def run_full_pipeline(self, clinical_df: pd.DataFrame, k_features: int = 30) -> Dict[str, Any]:
        # Compute text features
        tfidf_df = self.compute_tfidf()
        w2v_df = self.compute_word2vec()
        # Combine with clinical features (excluding text columns)
        clinical_features = clinical_df.drop(columns=[self.text_column, 'summary', 'tokens', 'lemmas', 'stems', 'no_stopwords', 'spacy_lemmas'], errors='ignore')
        combined = self.combine_features([clinical_features.reset_index(drop=True)])
        # Feature selection
        y = self.df[self.label_column]
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        X_selected = self.select_features(combined, y, k=k_features)
        # Feature importance
        top_features = self.feature_importance(X_selected, y)
        top_perm = self.permutation_importance(X_selected, y)
        return {
            'tfidf': tfidf_df,
            'word2vec': w2v_df,
            'combined': combined,
            'selected': X_selected,
            'top_features': top_features,
            'top_permutation': top_perm
        }

def main():
    from data_loader import AppointmentDataLoader
    from data_preprocessing import DataPreprocessor
    from clinical_feature_extraction import ClinicalFeatureExtractor
    # Load and preprocess data
    loader = AppointmentDataLoader()
    df = loader.to_dataframe()
    preprocessor = DataPreprocessor(df)
    processed_df = preprocessor.preprocess_data()
    # Extract clinical features
    extractor = ClinicalFeatureExtractor(processed_df)
    clinical_df = extractor.extract_all_features()
    # Text feature engineering
    engineer = TextFeatureEngineer(clinical_df)
    results = engineer.run_full_pipeline(clinical_df, k_features=30)
    print("\nTop 10 selected features:")
    print(results['selected'].columns[:10].tolist())
    print("\nTop 10 feature importances:")
    print(results['top_features'][:10])
    print("\nTop 10 permutation importances:")
    print(results['top_permutation'][:10])
    # Save selected features to CSV
    results['selected'].to_csv('text_feature_engineering_selected.csv', index=False)

if __name__ == "__main__":
    main() 