"""
DataProcessor class for handling data preprocessing and management in the sports analytics application.
This class handles data loading, preprocessing, and provides information about the dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Dict, List, Optional, Union

class DataProcessor:
    """
    A class to handle data processing operations including loading, preprocessing, and data management.
    
    Attributes:
        raw_data (pd.DataFrame): The original, unprocessed dataset
        processed_data (pd.DataFrame): The preprocessed dataset ready for analysis
        preprocessor (ColumnTransformer): The preprocessing pipeline for data transformation
        column_info (Dict): Dictionary storing information about each column
        exclude_from_encoding (List): List of column names to exclude from one-hot encoding
    """

    def __init__(self, exclude_columns=None):
        """
        Initialize the DataProcessor with optional columns to exclude from encoding.
        
        Args:
            exclude_columns (Union[str, List[str]], optional): Column(s) to exclude from one-hot encoding
        """
        self.raw_data = None
        self.processed_data = None
        self.preprocessor = None
        self.column_info = {}  # Store information about each column
        # Initialize empty list for excluded columns
        self.exclude_from_encoding = []
        if exclude_columns is not None:
            if isinstance(exclude_columns, str):
                exclude_columns = [exclude_columns]
            self.exclude_from_encoding.extend([col.lower() for col in exclude_columns])
            # Remove duplicates while preserving order
            self.exclude_from_encoding = list(dict.fromkeys(self.exclude_from_encoding))

    def update_excluded_columns(self, exclude_columns):
        """
        Update the list of columns to exclude from one-hot encoding.
        
        Args:
            exclude_columns (Union[str, List[str]]): Column(s) to exclude from one-hot encoding
        """
        if isinstance(exclude_columns, str):
            exclude_columns = [exclude_columns]
        # Clear existing excluded columns and add new ones
        self.exclude_from_encoding = [col.lower() for col in exclude_columns]
        # Remove duplicates while preserving order
        self.exclude_from_encoding = list(dict.fromkeys(self.exclude_from_encoding))
        # Reprocess data if it exists
        if self.raw_data is not None:
            self._process_data()

    def load_data(self, file_path):
        """
        Load data from a CSV file path or file-like object.
        
        Args:
            file_path: Path to CSV file or file-like object
            
        Returns:
            pd.DataFrame: The loaded raw data
            
        Raises:
            ValueError: If file is empty or loading fails
        """
        try:
            self.raw_data = pd.read_csv(file_path)
            if self.raw_data.empty:
                raise ValueError("The uploaded file is empty")
            
            # Store column information
            self._store_column_info()
            
            self._process_data()  # Process data immediately
            return self.raw_data
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    def _store_column_info(self) -> None:
        """
        Store detailed information about each column in the dataset.
        This includes data type, missing values, unique values, and statistical measures.
        """
        for col in self.raw_data.columns:
            self.column_info[col] = {
                'dtype': str(self.raw_data[col].dtype),
                'missing_values': self.raw_data[col].isnull().sum(),
                'unique_values': self.raw_data[col].nunique(),
                'min': self.raw_data[col].min() if pd.api.types.is_numeric_dtype(self.raw_data[col]) else None,
                'max': self.raw_data[col].max() if pd.api.types.is_numeric_dtype(self.raw_data[col]) else None,
                'mean': self.raw_data[col].mean() if pd.api.types.is_numeric_dtype(self.raw_data[col]) else None,
                'std': self.raw_data[col].std() if pd.api.types.is_numeric_dtype(self.raw_data[col]) else None
            }

    def _process_data(self):
        """
        Process raw data using standardization for numerical and one-hot encoding for categorical variables.
        Handles missing values and creates a preprocessing pipeline.
        
        Raises:
            ValueError: If processing fails or no valid columns are found
        """
        if self.raw_data is None:
            return

        try:
            # Define numerical/categorical columns
            num_cols = self.raw_data.select_dtypes(include=['number']).columns
            cat_cols = self.raw_data.select_dtypes(include=['object']).columns
            
            # Exclude specified columns from one-hot encoding
            cat_cols = [col for col in cat_cols if col.lower() not in self.exclude_from_encoding]

            # Handle missing values
            self.raw_data[num_cols] = self.raw_data[num_cols].fillna(self.raw_data[num_cols].mean())
            self.raw_data[cat_cols] = self.raw_data[cat_cols].fillna('Unknown')

            # Create preprocessing pipeline
            transformers = []
            if len(num_cols) > 0:
                transformers.append(('num', StandardScaler(), num_cols))
            if len(cat_cols) > 0:
                transformers.append(('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols))

            if not transformers:
                raise ValueError("No valid columns found for processing")

            self.preprocessor = ColumnTransformer(transformers)

            # Process data
            processed_array = self.preprocessor.fit_transform(self.raw_data)
            
            # Get feature names
            feature_names = []
            for name, transformer, cols in transformers:
                if name == 'num':
                    feature_names.extend(cols)
                elif name == 'cat':
                    cat_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
                    feature_names.extend(cat_features)

            # Create processed DataFrame
            self.processed_data = pd.DataFrame(processed_array, columns=feature_names)
            
        except Exception as e:
            raise ValueError(f"Error processing data: {str(e)}")

    def get_processed_columns(self):
        """
        Get the list of processed column names.
        
        Returns:
            List[str]: List of processed column names
        """
        if self.processed_data is not None:
            return self.processed_data.columns.tolist()
        return []

    def get_column_info(self):
        """
        Get information about the columns in the dataset.
        
        Returns:
            Dict: Dictionary containing information about each column
        """
        return self.column_info

    def get_data_summary(self) -> Dict:
        """
        Get a comprehensive summary of the dataset.
        
        Returns:
            Dict: Dictionary containing dataset summary information
        """
        if self.raw_data is None:
            return {}
            
        summary = {
            'shape': self.raw_data.shape,
            'missing_values': self.raw_data.isnull().sum().sum(),
            'duplicates': self.raw_data.duplicated().sum(),
            'numeric_columns': len(self.raw_data.select_dtypes(include=['number']).columns),
            'categorical_columns': len(self.raw_data.select_dtypes(include=['object']).columns),
            'date_columns': len(self.raw_data.select_dtypes(include=['datetime64']).columns),
            'column_info': self.column_info
        }
        
        return summary

    def get_excluded_columns(self):
        """
        Get the list of columns currently excluded from one-hot encoding.
        
        Returns:
            List[str]: List of excluded column names
        """
        return self.exclude_from_encoding