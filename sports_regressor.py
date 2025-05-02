from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class SportsRegressor:
    def __init__(self):
        self.model = None
        self.metrics = {}
        # Add Random Forest to model types
        self.model_types = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_name = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def prepare_data(self, data, target_column):
        """Prepare data for training with validation checks"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
            
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        if data.empty:
            raise ValueError("Data is empty")
            
        # Check for missing values in target
        if data[target_column].isnull().any():
            raise ValueError(f"Target column '{target_column}' contains missing values")
            
        # Store feature and target names
        self.feature_names = data.drop(columns=[target_column]).columns.tolist()
        self.target_name = target_column
        
        # Split features and target
        features = data.drop(columns=[target_column])
        target = data[target_column]

        # Validate features
        if features.empty:
            raise ValueError("No features available for training")
            
        # Check for constant columns
        constant_cols = features.columns[features.nunique() == 1]
        if not constant_cols.empty:
            raise ValueError(f"Constant columns found: {', '.join(constant_cols)}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        # Scale features and target
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        y_train_scaled = self.target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_scaled = self.target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

        # Store scaled data
        self.X_train = pd.DataFrame(X_train_scaled, columns=features.columns)
        self.X_test = pd.DataFrame(X_test_scaled, columns=features.columns)
        self.y_train = pd.Series(y_train_scaled)
        self.y_test = pd.Series(y_test_scaled)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self, model_type='linear'):
        """Train the model with validation checks"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data has not been prepared. Call prepare_data() first")
            
        if model_type not in self.model_types:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        try:
            self.model = self.model_types[model_type]
            self.model.fit(self.X_train, self.y_train)
        except Exception as e:
            raise ValueError(f"Error training model: {str(e)}")

    def evaluate(self):
        """Evaluate the model with comprehensive metrics"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        try:
            # Make predictions on scaled data
            y_pred_scaled = self.model.predict(self.X_test)
            
            # Inverse transform predictions and actual values
            y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            y_test = self.target_scaler.inverse_transform(self.y_test.values.reshape(-1, 1)).ravel()
            
            # Calculate metrics on original scale
            self.metrics = {
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R²': r2_score(y_test, y_pred),
                'Explained Variance': self.model.score(self.X_test, self.y_test)
            }
            
            # Add feature importance based on model type
            if isinstance(self.model, LinearRegression):
                # Calculate standardized coefficients for linear regression
                std_coef = self.model.coef_ * np.std(self.X_train, axis=0)
                self.metrics['Feature Importance'] = dict(zip(
                    self.feature_names,
                    std_coef
                ))
            elif isinstance(self.model, RandomForestRegressor):
                # Get feature importance from random forest
                importance = self.model.feature_importances_
                self.metrics['Feature Importance'] = dict(zip(
                    self.feature_names,
                    importance
                ))
                
            return self.metrics
        except Exception as e:
            raise ValueError(f"Error evaluating model: {str(e)}")

    def predict(self, X_new):
        """Make predictions with validation checks"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        if not isinstance(X_new, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        if not all(col in X_new.columns for col in self.feature_names):
            raise ValueError("Input data missing required features")
            
        try:
            # Scale the input features
            X_new_scaled = self.feature_scaler.transform(X_new[self.feature_names])
            
            # Make predictions on scaled data
            y_pred_scaled = self.model.predict(X_new_scaled)
            
            # Inverse transform predictions to original scale
            y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            return y_pred
        except Exception as e:
            raise ValueError(f"Error making predictions: {str(e)}")