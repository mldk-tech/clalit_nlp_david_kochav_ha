import pandas as pd
from typing import List, Dict, Any
import logging
from data_loader import AppointmentDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OutcomeLabeler:
    """
    Handles outcome keyword definition, mapping, and validation for binary classification.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.positive_keywords = [
            "patient responded positively to treatment",
            "health has returned to baseline",
            "feeling much better",
            "recovery is on track",
            "no complaints at follow-up",
            "no further issues reported",
            "doing well",
            "symptoms have subsided",
            "recovery is complete",
            "excellent recovery progress",
            "patient discharged in good condition"
        ]
        self.negative_keywords = [
            "condition has deteriorated",
            "patient deceased",
            "critical condition",
            "despite efforts, the patient passed away",
            "emergency intervention required",
            "symptoms worsened",
            "patient is in critical condition",
            "patient admitted for further evaluation",
            "no improvement noted",
            "health continues to decline",
            "state declined",
            "alternative treatment considered"
        ]
        self.neutral_keywords = [
            "condition remains unchanged",
            "symptoms are persistent but manageable",
            "stable condition",
            "patient reports minor improvements but overall stable",
            "no significant progress, further observation required",
            "monitoring continues"
        ]

    def label_outcome(self, outcome: str) -> str:
        outcome_lower = outcome.lower()
        for kw in self.positive_keywords:
            if kw in outcome_lower:
                return "better"
        for kw in self.negative_keywords:
            if kw in outcome_lower:
                return "worse"
        for kw in self.neutral_keywords:
            if kw in outcome_lower:
                return "better"  # Treat neutral as 'better' for binary, or could be a third class
        return "worse"  # Default to 'worse' if uncategorized

    def apply_labeling(self) -> pd.DataFrame:
        self.df['outcome_label'] = self.df['future_outcome'].apply(self.label_outcome)
        return self.df

    def validate_labeling(self) -> Dict[str, Any]:
        labeled_df = self.apply_labeling()
        counts = labeled_df['outcome_label'].value_counts().to_dict()
        unique_outcomes = labeled_df['future_outcome'].nunique()
        total = len(labeled_df)
        logger.info(f"Outcome labeling: {counts}")
        return {
            "label_counts": counts,
            "unique_outcomes": unique_outcomes,
            "total_records": total,
            "label_distribution": {k: v/total for k, v in counts.items()}
        }

    def get_keywords(self) -> Dict[str, List[str]]:
        return {
            "positive": self.positive_keywords,
            "negative": self.negative_keywords,
            "neutral": self.neutral_keywords
        }

def main():
    loader = AppointmentDataLoader()
    df = loader.to_dataframe()
    labeler = OutcomeLabeler(df)
    labeled_df = labeler.apply_labeling()
    validation = labeler.validate_labeling()
    print("=== Outcome Labeling Validation ===")
    for k, v in validation.items():
        print(f"{k}: {v}")
    print("\nSample labeled data:")
    print(labeled_df[['future_outcome', 'outcome_label']].head(10))
    # Save to CSV
    labeled_df[['future_outcome', 'outcome_label']].to_csv('outcome_labeling.csv', index=False)

if __name__ == "__main__":
    main() 