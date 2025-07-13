import json
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppointmentDataLoader:
    """
    Data loader for doctor appointment summaries dataset.
    """
    
    def __init__(self, data_path: str = "../doctor_appointment_summaries.json"):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the JSON dataset file
        """
        self.data_path = Path(data_path)
        self.data = None
        self.df = None
        
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load the JSON dataset.
        
        Returns:
            List of appointment records
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            with open(self.data_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
            
            logger.info(f"Successfully loaded {len(self.data)} appointment records")
            return self.data
            
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the loaded data to a pandas DataFrame.
        
        Returns:
            DataFrame with appointment data
        """
        if self.data is None:
            self.load_data()
        
        self.df = pd.DataFrame(self.data)
        logger.info(f"Converted data to DataFrame with shape: {self.df.shape}")
        return self.df
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics about the dataset.
        
        Returns:
            Dictionary with basic statistics
        """
        if self.df is None:
            self.df = self.to_dataframe()
        
        stats = {
            'total_records': len(self.df),
            'columns': list(self.df.columns),
            'data_types': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'unique_doctors': self.df['doctor_id'].nunique(),
            'doctor_distribution': self.df['doctor_id'].value_counts().to_dict(),
            'summary_length_stats': {
                'mean': self.df['summary'].str.len().mean(),
                'median': self.df['summary'].str.len().median(),
                'min': self.df['summary'].str.len().min(),
                'max': self.df['summary'].str.len().max(),
                'std': self.df['summary'].str.len().std()
            },
            'unique_outcomes': self.df['future_outcome'].nunique()
        }
        
        return stats
    
    def validate_data_structure(self) -> Dict[str, bool]:
        """
        Validate the data structure and completeness.
        
        Returns:
            Dictionary with validation results
        """
        if self.df is None:
            self.df = self.to_dataframe()
        
        validation_results = {
            'has_required_columns': all(col in self.df.columns for col in ['id', 'doctor_id', 'summary', 'future_outcome']),
            'no_missing_values': not self.df.isnull().values.any(),
            'valid_uuid_format': self.df['id'].str.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$').all(),
            'valid_doctor_format': self.df['doctor_id'].str.match(r'^dr_\d{2}$').all(),
            'non_empty_summaries': (self.df['summary'].str.len() > 0).all(),
            'non_empty_outcomes': (self.df['future_outcome'].str.len() > 0).all()
        }
        
        return validation_results

def main():
    """
    Main function to demonstrate data loading and validation.
    """
    # Initialize data loader
    loader = AppointmentDataLoader()
    
    # Load data
    data = loader.load_data()
    
    # Convert to DataFrame
    df = loader.to_dataframe()
    
    # Get basic statistics
    stats = loader.get_basic_stats()
    
    # Validate data structure
    validation = loader.validate_data_structure()
    
    # Print results
    print("=== Dataset Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== Data Validation ===")
    for key, value in validation.items():
        status = "✓" if value else "✗"
        print(f"{status} {key}: {value}")
    
    print(f"\n=== Sample Data ===")
    print(df.head())
    
    return loader

if __name__ == "__main__":
    main() 