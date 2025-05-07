"""
SportsChatbot class for providing interactive data analysis and model interpretation.
This class serves as an AI assistant for the sports analytics application, helping users understand their data and models.
"""

from data_processor import DataProcessor  
from sports_regressor import SportsRegressor

class SportsChatbot:
    """
    A class to provide interactive assistance for sports data analysis.
    
    This chatbot can:
    - Answer questions about the dataset
    - Explain model results and metrics
    - Provide visualization guidance
    - Offer data quality insights
    
    Attributes:
        data_processor (DataProcessor): Instance for data processing operations
        model (SportsRegressor): Instance for regression modeling
        last_metrics (Dict): Most recent model metrics
        column_info (Dict): Information about dataset columns
        faq (Dict): Dictionary of frequently asked questions and their responses
    """

    def __init__(self):
        """Initialize the SportsChatbot with default settings and FAQ responses."""
        self.data_processor = None
        self.model = None
        self.last_metrics = {}
        self.column_info = {}

        self.faq = {
            "normalization": "Normalization scales features to have zero mean and unit variance, making models converge faster and perform better. This helps prevent features with larger scales from dominating the model.",
            "missing values": "Missing values are handled using imputation. For numerical columns, we fill missing values with the mean. For categorical columns, we use 'Unknown' as a placeholder. This ensures the model can work with incomplete data.",
            "feature engineering": "Feature engineering creates new features by combining or transforming existing ones to help the model capture patterns better. This can include creating interaction terms, polynomial features, or domain-specific transformations.",
            "data summary": self.get_data_summary,
            "model accuracy": self.get_model_accuracy,
            "help": "Try asking about: 'data summary', 'model accuracy', 'visualization help', or ask about preprocessing like 'missing values' or 'normalization'.",
            "visualization help": """Here's how to use the visualization features:

1. Chart Types:
   - Scatter Plot: Best for showing relationships between two numerical variables
   - Line Chart: Great for showing trends over time or ordered categories
   - Bar Chart: Perfect for comparing categorical data

2. Color Schemes:
   - No Color: Basic visualization without color encoding
   - Viridis: A perceptually uniform color scheme, good for continuous data
   - Plasma: A high-contrast color scheme, good for highlighting differences
   - Inferno: A dark-to-light color scheme, good for heatmap-like visualizations
   - Magma: A dark-to-light color scheme with purple tones
   - Cividis: A colorblind-friendly color scheme
   - Rainbow: A traditional rainbow color scheme
   - Jet: A classic color scheme with high contrast
   - Hot: A black-red-yellow color scheme, good for intensity
   - Cool: A cyan-magenta color scheme

3. How to Use:
   - Select your X-axis and Y-axis variables
   - Choose a chart type that best represents your data
   - Select a color scheme to enhance your visualization
   - Click 'Update Chart' to see your visualization

4. Tips:
   - Use scatter plots to find correlations between variables
   - Use line charts for time series or sequential data
   - Use bar charts for comparing categories
   - Try different color schemes to find the most effective visualization
   - The colorbar shows the scale of your Y-axis values

- "What does the colorbar show?" """,
            "model help": "For model help, consider these points:\n1. Choose a target variable that you want to predict\n2. Select relevant features that might influence the target\n3. Check model metrics to evaluate performance",
            "data quality": self.get_data_quality_report
        }

    def connect_to_data(self, data_processor):
        """
        Connect the chatbot to a DataProcessor instance.
        
        Args:
            data_processor (DataProcessor): Instance of DataProcessor to connect to
            
        Raises:
            ValueError: If data_processor is not a DataProcessor instance
        """
        if not isinstance(data_processor, DataProcessor):
            raise ValueError("data_processor must be an instance of DataProcessor")
        self.data_processor = data_processor
        self.column_info = data_processor.get_column_info()

    def connect_to_model(self, model):
        """
        Connect the chatbot to a SportsRegressor instance.
        
        Args:
            model (SportsRegressor): Instance of SportsRegressor to connect to
            
        Raises:
            ValueError: If model is not a SportsRegressor instance
        """
        if not isinstance(model, SportsRegressor):
            raise ValueError("model must be an instance of SportsRegressor")
        self.model = model

    def get_response(self, user_input):
        """
        Get a response to user input by analyzing the query and providing relevant information.
        
        Args:
            user_input (str): The user's question or request
            
        Returns:
            str: Response to the user's input
        """
        user_input = user_input.lower().strip()

        # Add error handling for missing data/model
        if "data" in user_input or "dataset" in user_input:
            if self.data_processor is None or self.data_processor.raw_data is None:
                return "Please upload data first!"
            
        if "model" in user_input or "accuracy" in user_input or "metrics" in user_input:
            if self.model is None or not hasattr(self.model, 'metrics'):
                return "No trained model exists yet!"
        
        # Check for FAQ matches first
        for key in self.faq:
            if key in user_input:
                if callable(self.faq[key]):
                    return self.faq[key]()
                return self.faq[key]
        
        # Data-related queries
        if "data" in user_input or "dataset" in user_input:
            return self.get_data_summary()
        
        # Model-related queries
        if "model" in user_input or "accuracy" in user_input or "metrics" in user_input:
            return self.get_model_accuracy()
            
        # Visualization queries
        if any(term in user_input for term in ["visual", "plot", "chart", "graph", "show me", "display", "visualization"]):
            return self.faq["visualization help"]
            
        # Data quality queries
        if "quality" in user_input or "missing" in user_input:
            return self.get_data_quality_report()
            
        # Help queries
        if "help" in user_input:
            return self.faq["help"]
        
        return "I'm not sure how to help with that. Try asking about your data or model, or type 'help' for suggestions."

    def get_data_summary(self):
        """
        Get a summary of the current dataset.
        
        Returns:
            str: Formatted summary of the dataset including size and column information
        """
        if self.data_processor is None or self.data_processor.raw_data is None:
            return "No data loaded yet. Please upload a CSV file first."
        
        df = self.data_processor.raw_data
        summary = [
            f"Your dataset has {len(df)} rows and {len(df.columns)} columns.",
            "\nColumns and their types:"
        ]
        
        for col, info in self.column_info.items():
            summary.append(
                f"- {col}: {info['dtype']} (Missing: {info['missing_values']}, "
                f"Unique: {info['unique_values']})"
            )
            
        return "\n".join(summary)

    def get_model_accuracy(self):
        """
        Get the current model's accuracy metrics.
        
        Returns:
            str: Formatted string containing model performance metrics
        """
        if self.model is None or not hasattr(self.model, 'metrics') or not self.model.metrics:
            return "No model trained yet. Please train a model first."
        
        metrics = self.model.metrics
        response = [
            "Model Performance Metrics:",
            f"R² Score: {metrics.get('R²', 0):.4f} (closer to 1 is better)",
            f"RMSE: {metrics.get('RMSE', 0):.4f} (lower is better)",
            f"MAE: {metrics.get('MAE', 0):.4f} (lower is better)",
            f"Explained Variance: {metrics.get('Explained Variance', 0):.4f}"
        ]
        
        return "\n".join(response)
    
    def get_data_quality_report(self):
        """
        Get a comprehensive data quality report.
        
        Returns:
            str: Formatted string containing data quality information
        """
        if self.data_processor is None or self.data_processor.raw_data is None:
            return "No data loaded yet. Please upload a CSV file first."
            
        df = self.data_processor.raw_data
        report = ["Data Quality Report:"]
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            report.append("\nMissing Values:")
            for col, count in missing_values[missing_values > 0].items():
                percentage = (count / len(df)) * 100
                report.append(f"- {col}: {count} missing values ({percentage:.1f}%)")
        else:
            report.append("\nNo missing values found in the dataset.")
            
        # Check for constant columns
        constant_cols = df.columns[df.nunique() == 1]
        if not constant_cols.empty:
            report.append("\nConstant Columns (might need attention):")
            for col in constant_cols:
                report.append(f"- {col}")
                
        # Check for data types
        report.append("\nData Types:")
        for col, info in self.column_info.items():
            report.append(f"- {col}: {info['dtype']}")
            
        return "\n".join(report)
    
    def interpret_metrics(self, metrics):
        """
        Interpret model metrics in a user-friendly way.
        
        Args:
            metrics (Dict): Dictionary containing model metrics
            
        Returns:
            str: User-friendly interpretation of the metrics
        """
        self.last_metrics = metrics
        messages = []
        
        if 'R²' in metrics:
            r2 = metrics['R²']
            messages.append(f"Your R² score of {r2:.4f} means the model explains {r2*100:.1f}% of the variance in the target variable.")
            
        if 'RMSE' in metrics:
            rmse = metrics['RMSE']
            messages.append(f"Your RMSE of {rmse:.4f} suggests the predictions are off by about {rmse:.4f} units on average.")
            
        if 'MAE' in metrics:
            mae = metrics['MAE']
            messages.append(f"The MAE of {mae:.4f} indicates an average absolute error of {mae:.4f} units.")
            
        return " ".join(messages) if messages else "No metrics available."