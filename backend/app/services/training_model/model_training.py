import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import logging
import csv
import re
import io
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import joblib
from backend.app.services.model_metrics_service import ModelMetricsService
from backend.app.db.session import get_db

from data_loader import AppointmentDataLoader
from data_preprocessing import DataPreprocessor
from clinical_feature_extraction import ClinicalFeatureExtractor
from text_feature_engineering import TextFeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    logger.warning('XGBoost is not installed. Please install xgboost to use XGBClassifier.')

# Utility to ensure only 1D numeric columns are used

def filter_numeric_1d(df):
    return df[[col for col in df.columns if np.issubdtype(df[col].dtypes, np.number) and getattr(df[col], 'ndim', 1) == 1]]

def train_random_forest(X, y, param_grid=None, cv=5):
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring='f1', n_jobs=-1)
    grid.fit(X, y)
    logger.info(f'Best Random Forest params: {grid.best_params_}')
    logger.info(f'Best cross-validated F1: {grid.best_score_:.4f}')
    return grid.best_estimator_, grid.cv_results_

def train_xgboost(X, y, param_grid=None, cv=5):
    if XGBClassifier is None:
        raise ImportError('XGBoost is not installed. Please install xgboost to use XGBClassifier.')
    # Ensure only 1D numeric columns are passed to XGBoost
    X = filter_numeric_1d(X)
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    grid = GridSearchCV(xgb, param_grid, cv=cv, scoring='f1', n_jobs=-1)
    grid.fit(X, y)
    logger.info(f'Best XGBoost params: {grid.best_params_}')
    logger.info(f'Best cross-validated F1: {grid.best_score_:.4f}')
    return grid.best_estimator_, grid.cv_results_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, digits=3)
    }
    return metrics

# Function to save output to CSVs

def save_xgboost_output_to_csv(output_str):
    # 1. Extract metrics
    metrics = {}
    metrics_section = re.search(r'XGBoost Evaluation Metrics:(.*?)confusion_matrix:', output_str, re.DOTALL)
    if metrics_section:
        for line in metrics_section.group(1).strip().split('\n'):
            if ':' in line:
                k, v = line.split(':', 1)
                metrics[k.strip()] = v.strip()
    with open('xgboost_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for k, v in metrics.items():
            writer.writerow([k, v])

    # 2. Extract feature importances
    importances = []
    imp_section = re.search(r'Top 10 XGBoost Feature Importances:(.*?)(?:\n\n|$)', output_str, re.DOTALL)
    if imp_section:
        for line in imp_section.group(1).strip().split('\n'):
            if ':' in line:
                feat, val = line.split(':')
                importances.append([feat.strip(), val.strip()])
    with open('xgboost_feature_importances.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature', 'Importance'])
        writer.writerows(importances)

    # 3. Extract model comparison table
    comp_section = re.search(r'=== Baseline Model Comparison Summary ===(.*?)Review the above', output_str, re.DOTALL)
    if comp_section:
        lines = [l for l in comp_section.group(1).strip().split('\n') if l.strip() and not l.startswith('-')]
        header = [h.strip() for h in lines[0].split('|')]
        rows = []
        for line in lines[1:]:
            if 'Best:' in line:
                line = line.split('<--')[0]
            row = [c.strip() for c in line.split('|')]
            if len(row) == len(header):
                rows.append(row)
        with open('model_comparison.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

def plot_confusion_matrix(cm, model_name, save_path=None):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_proba, model_name, save_path=None):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend(loc='lower right')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_feature_importance(importances, feature_names, model_name, save_path=None):
    idx = np.argsort(importances)[::-1][:10]
    plt.figure(figsize=(8,5))
    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx])
    plt.title(f'Top 10 Feature Importances: {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def statistical_significance_test(y_true, pred1, pred2, model1_name, model2_name):
    # Paired t-test on predictions (binary)
    t_stat, p_val = ttest_rel(pred1, pred2)
    print(f"Statistical significance test between {model1_name} and {model2_name}: t={t_stat:.4f}, p={p_val:.4f}")
    return t_stat, p_val

def save_metrics_to_csv(metrics_dict, filename):
    df = pd.DataFrame(metrics_dict).T
    df.to_csv(filename)

def run_training_pipeline():
    """
    Runs the full model training pipeline and returns a summary of results.
    """
    # Load and preprocess data
    loader = AppointmentDataLoader()
    df = loader.to_dataframe()
    preprocessor = DataPreprocessor(df)
    processed_df = preprocessor.preprocess_data()
    
    # Extract clinical features
    extractor = ClinicalFeatureExtractor(processed_df)
    clinical_df = extractor.extract_all_features()
    
    # Feature engineering
    engineer = TextFeatureEngineer(clinical_df)
    features = engineer.run_full_pipeline(clinical_df, k_features=30)
    
    # CRITICAL FIX: Remove target variable from features to prevent data leakage
    if 'outcome_encoded' in features['selected'].columns:
        features['selected'] = features['selected'].drop('outcome_encoded', axis=1)
        logger.warning("Removed 'outcome_encoded' from features to prevent data leakage")
    
    X = features['selected']
    y = processed_df['outcome_encoded']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=X.columns)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=X.columns)
    
    output_buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = output_buffer
    
    # Initialize database service
    db = next(get_db())
    metrics_service = ModelMetricsService(db)
    
    try:
        model_results = {}
        # Train Random Forest
        best_rf, rf_cv_results = train_random_forest(X_train, y_train)
        rf_metrics = evaluate_model(best_rf, X_test, y_test)
        model_results['RandomForest'] = rf_metrics
        
        # Save Random Forest metrics to database
        rf_metrics_dict = {
            'accuracy': rf_metrics['accuracy'],
            'precision': rf_metrics['precision'],
            'recall': rf_metrics['recall'],
            'f1': rf_metrics['f1'],
            'roc_auc': rf_metrics['roc_auc'] if rf_metrics['roc_auc'] else 0.0
        }
        metrics_service.save_model_metrics('random_forest_v1', rf_metrics_dict)
        
        # Save Random Forest feature importances
        rf_importances = dict(zip(X.columns, best_rf.feature_importances_))
        metrics_service.save_feature_importances('random_forest_v1', rf_importances)
        
        save_metrics_to_csv({'RandomForest': rf_metrics}, 'rf_metrics.csv')
        plot_confusion_matrix(rf_metrics['confusion_matrix'], 'RandomForest', 'rf_confusion_matrix.png')
        if rf_metrics['roc_auc'] is not None:
            y_prob_rf = best_rf.predict_proba(X_test)[:,1]
            plot_roc_curve(y_test, y_prob_rf, 'RandomForest', 'rf_roc_curve.png')
        plot_feature_importance(best_rf.feature_importances_, X.columns, 'RandomForest', 'rf_feature_importance.png')
        # Save Random Forest model
        joblib.dump(best_rf, 'results/random_forest_model.pkl')
        
        # Train XGBoost
        xgb_metrics = None
        if XGBClassifier is not None:
            X_train_xgb = filter_numeric_1d(X_train)
            X_test_xgb = filter_numeric_1d(X_test)
            if not isinstance(X_test_xgb, pd.DataFrame):
                X_test_xgb = pd.DataFrame(X_test_xgb)
            X_test_xgb = X_test_xgb.reindex(columns=X_train_xgb.columns, fill_value=0)
            best_xgb, xgb_cv_results = train_xgboost(X_train_xgb, y_train)
            xgb_metrics = evaluate_model(best_xgb, X_test_xgb, y_test)
            model_results['XGBoost'] = xgb_metrics
            
            # Save XGBoost metrics to database
            xgb_metrics_dict = {
                'accuracy': xgb_metrics['accuracy'],
                'precision': xgb_metrics['precision'],
                'recall': xgb_metrics['recall'],
                'f1': xgb_metrics['f1'],
                'roc_auc': xgb_metrics['roc_auc'] if xgb_metrics['roc_auc'] else 0.0
            }
            metrics_service.save_model_metrics('xgboost_v1', xgb_metrics_dict)
            
            # Save XGBoost feature importances
            xgb_importances = dict(zip(X_train_xgb.columns, best_xgb.feature_importances_))
            metrics_service.save_feature_importances('xgboost_v1', xgb_importances)
            
            save_metrics_to_csv({'XGBoost': xgb_metrics}, 'xgb_metrics.csv')
            plot_confusion_matrix(xgb_metrics['confusion_matrix'], 'XGBoost', 'xgb_confusion_matrix.png')
            if xgb_metrics['roc_auc'] is not None:
                y_prob_xgb = best_xgb.predict_proba(X_test_xgb)[:,1]
                plot_roc_curve(y_test, y_prob_xgb, 'XGBoost', 'xgb_roc_curve.png')
            plot_feature_importance(best_xgb.feature_importances_, X_train_xgb.columns, 'XGBoost', 'xgb_feature_importance.png')
            # Save XGBoost model
            joblib.dump(best_xgb, 'results/xgboost_model.pkl')
        
        # Model comparison
        comparison = {
            'accuracy': {k: v['accuracy'] for k, v in model_results.items()},
            'precision': {k: v['precision'] for k, v in model_results.items()},
            'recall': {k: v['recall'] for k, v in model_results.items()},
            'f1': {k: v['f1'] for k, v in model_results.items()},
            'roc_auc': {k: v['roc_auc'] for k, v in model_results.items()}
        }
        
        # Save model comparison to database
        metrics_service.save_model_comparison(comparison)
        
        save_metrics_to_csv(comparison, 'model_comparison.csv')
        if xgb_metrics is not None:
            t_stat, p_val = statistical_significance_test(y_test, best_rf.predict(X_test), best_xgb.predict(X_test_xgb), 'RandomForest', 'XGBoost')
            with open('model_significance_test.txt', 'w') as f:
                f.write(f"t-statistic: {t_stat}\np-value: {p_val}\n")
        print("\nAll evaluation metrics, plots, and comparison tables have been saved.")
    finally:
        sys.stdout = sys_stdout
        db.close()
    
    output_str = output_buffer.getvalue()
    save_xgboost_output_to_csv(output_str)
    # Return a summary dictionary
    return {
        'rf_metrics': rf_metrics,
        'xgb_metrics': xgb_metrics if XGBClassifier is not None else None,
        'comparison': comparison,
        'output_log': output_str
    }

def main():
    run_training_pipeline()

if __name__ == "__main__":
    main() 