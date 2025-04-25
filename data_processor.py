import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler #normalizes numerical data (mean = 0, std = 1).
from sklearn.impute import SimpleImputer #fills missing values (e.g., replacing NaNs with mean).

class SimpleDataProcessor:  
    def __init__(self):   #Constructor: initializes an empty variable data that will store the loaded dataset.
        self.data = None

    def load_data(self, file_path):
        
        self.data = pd.read_csv(file_path)
        return self.data

    def process(self):      #Handle missing values and normalize numerical data
        
        if self.data is None:
            return None

        num_cols = self.data.select_dtypes(include=['int64', 'float64']).columns    # all numerical columns (integer or float).
        cat_cols = self.data.select_dtypes(include=['object']).columns              # all categorical columns (strings/objects).

        # Handle missing values
        self.data[num_cols] = SimpleImputer(strategy='mean').fit_transform(self.data[num_cols])     #Fill missing values in numerical columns with the mean of each column.
        self.data[cat_cols] = self.data[cat_cols].fillna('Unknown')                                 #fit_transform learns the means and replaces NaNs.

        # Normalize numerical columns
        self.data[num_cols] = StandardScaler().fit_transform(self.data[num_cols])  #Converts values so each column has a mean of 0 and standard deviation of 1.

        return self.data