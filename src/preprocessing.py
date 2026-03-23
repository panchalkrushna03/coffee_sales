import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pickle
import yaml

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CoffeeDataPreprocessor:
    """
    Preprocessor class for coffee sales data.
    Handles data cleaning, feature engineering, and transformation.
    """
    
    def __init__(self, config_path="config.yaml"):
        """Initialize preprocessor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scaler = None
        self.encoders = {}
        self.numeric_features = None
        self.categorical_features = None
        self.column_transformer = None
        
    def load_data(self, data_path):
        """Load raw data from CSV file."""
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Data shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        logger.info("Handling missing values")
        
        # Display missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Missing values:\n{missing_counts[missing_counts > 0]}")
        
        # Drop rows with missing target variable
        if self.config['preprocessing']['target_variable'] in df.columns:
            df = df.dropna(subset=[self.config['preprocessing']['target_variable']])
        
        # Drop rows with critical missing values
        df = df.dropna(subset=['unit_price', 'quantity', 'city', 'store_type', 'product_category'])
        
        # Fill other missing categorical values with 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna('Unknown', inplace=True)
        
        # Fill numeric missing values with median
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        logger.info(f"After handling missing values: {df.shape}")
        return df
    
    def feature_engineering(self, df):
        """Create engineered features."""
        logger.info("Performing feature engineering")
        
        # Convert timestamp to datetime and extract features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['month'] = df['timestamp'].dt.month
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['hour'] = df['timestamp'].dt.hour
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df = df.drop('timestamp', axis=1)
        
        # Convert discount_applied to numeric
        if 'discount_applied' in df.columns:
            df['discount_applied'] = (df['discount_applied'] == True).astype(int)
        
        # Convert loyalty_member to numeric
        if 'loyalty_member' in df.columns:
            df['loyalty_member'] = (df['loyalty_member'] == True).astype(int)
        
        # Create price-to-quantity ratio
        if 'total_amount' in df.columns and 'quantity' in df.columns:
            df['price_per_unit'] = df['total_amount'] / df['quantity']
        
        # Drop unnecessary columns
        cols_to_drop = ['transaction_id', 'customer_id', 'total_amount']
        df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
        
        logger.info(f"After feature engineering: {df.shape}")
        logger.info(f"Features: {df.columns.tolist()}")
        return df
    
    def prepare_features_and_target(self, df):
        """Separate features and target variable."""
        target_col = self.config['preprocessing']['target_variable']
        
        # Ensure target column exists
        if target_col not in df.columns:
            raise ValueError(f"Target variable '{target_col}' not found in dataset")
        
        y = df[[target_col]]
        X = df.drop(target_col, axis=1)
        
        self.numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        logger.info(f"Numeric features: {self.numeric_features}")
        logger.info(f"Categorical features: {self.categorical_features}")
        
        return X, y
    
    def build_preprocessor(self):
        """Build preprocessing pipeline."""
        logger.info("Building preprocessing pipeline")
        
        # Create transformers
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine transformers
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        
        return self.column_transformer
    
    def fit_and_transform(self, X):
        """Fit preprocessor and transform features."""
        logger.info("Fitting preprocessor on training data")
        X_transformed = self.column_transformer.fit_transform(X)
        return X_transformed
    
    def transform(self, X):
        """Transform features using fitted preprocessor."""
        X_transformed = self.column_transformer.transform(X)
        return X_transformed
    
    def preprocess(self, data_path):
        """
        Execute full preprocessing pipeline.
        Returns processed features and target variable.
        """
        logger.info("Starting preprocessing pipeline")
        
        # Load and clean data
        df = self.load_data(data_path)
        df = self.handle_missing_values(df)
        df = self.feature_engineering(df)
        
        # Prepare features and target
        X, y = self.prepare_features_and_target(df)
        
        # Build and fit preprocessor
        self.build_preprocessor()
        X_transformed = self.fit_and_transform(X)
        
        logger.info(f"Preprocessing complete. Final shape: {X_transformed.shape}")
        
        return X_transformed, y, X.columns.tolist()
    
    def save_preprocessor(self, filepath):
        """Save preprocessor to pickle file."""
        logger.info(f"Saving preprocessor to {filepath}")
        with open(filepath, 'wb') as f:
            pickle.dump(self.column_transformer, f)
    
    def load_preprocessor(self, filepath):
        """Load preprocessor from pickle file."""
        logger.info(f"Loading preprocessor from {filepath}")
        with open(filepath, 'rb') as f:
            self.column_transformer = pickle.load(f)