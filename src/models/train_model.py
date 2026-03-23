import logging
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CoffeeModelTrainer:
    """
    Trainer class for coffee sales price prediction model.
    Handles model training, evaluation, and saving.
    """
    
    def __init__(self, config_path="config.yaml"):
        """Initialize trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # initialize MLflow tracking
        mlflow_tracking_uri = self.config.get('mlflow', {}).get('tracking_uri', 'file:./mlruns')
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        self.model = None
        self.metrics = {}
    
    def load_data(self, train_path, test_path):
        """Load preprocessed train and test data."""
        logger.info(f"Loading train data from {train_path}")
        train_data = pd.read_csv(train_path)
        
        logger.info(f"Loading test data from {test_path}")
        test_data = pd.read_csv(test_path)
        
        # Separate features and target
        target_col_idx = -1  # Last column is the target
        
        X_train = train_data.iloc[:, :target_col_idx]
        y_train = train_data.iloc[:, target_col_idx]
        
        X_test = test_data.iloc[:, :target_col_idx]
        y_test = test_data.iloc[:, target_col_idx]
        
        logger.info(f"Train set shape: {X_train.shape}, {y_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}, {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self):
        """Build the machine learning model."""
        model_type = self.config['model']['type']
        hyperparams = self.config['model']['hyperparameters']
        
        logger.info(f"Building {model_type} model with hyperparameters: {hyperparams}")
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(**hyperparams)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(**hyperparams)
        elif model_type == 'linear_regression':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return self.model
    
    def train(self, X_train, y_train):
        """Train the model."""
        logger.info("Training model...")
        
        # Flatten y_train if needed
        y_train = np.ravel(y_train)
        
        self.model.fit(X_train, y_train)
        
        logger.info("Model training completed!")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        logger.info("Evaluating model...")
        
        # Flatten y_test if needed
        y_test = np.ravel(y_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Store metrics
        self.metrics = {
            'r2_score': float(r2),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae)
        }
        
        logger.info("=" * 50)
        logger.info("MODEL EVALUATION METRICS")
        logger.info("=" * 50)
        logger.info(f"R² Score: {r2:.4f}")
        logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
        logger.info("=" * 50)
        
        return self.metrics
    
    def save_model(self, filepath):
        """Save trained model to pickle file."""
        logger.info(f"Saving model to {filepath}")
        
        # Create directories if they don't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info("Model saved successfully!")
    
    def save_metrics(self, filepath):
        """Save metrics to JSON file."""
        logger.info(f"Saving metrics to {filepath}")
        
        # Create directories if they don't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        logger.info("Metrics saved successfully!")
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Complete training and evaluation pipeline with MLflow tracking."""
        logger.info("Starting training and evaluation pipeline")

        # Track experiment in MLflow
        mlflow.set_experiment(self.config.get('mlflow', {}).get('experiment_name', 'coffee_price_prediction'))

        with mlflow.start_run(run_name=self.config.get('mlflow', {}).get('run_name', 'coffee_price_run')):
            # Log model configuration and hyperparameters
            mlflow.log_params({
                'model_type': self.config['model']['type'],
                **self.config['model']['hyperparameters']
            })
            mlflow.log_params({
                'test_size': self.config['preprocessing']['test_size'],
                'random_state': self.config['preprocessing']['random_state']
            })

            # Build model
            self.build_model()

            # Train model
            self.train(X_train, y_train)

            # Evaluate model
            self.evaluate(X_test, y_test)

            # Log metrics
            mlflow.log_metrics(self.metrics)

            # Save temporary model artifact
            model_artifact_path = 'mlflow_model.pkl'
            self.save_model(model_artifact_path)
            mlflow.log_artifact(model_artifact_path, artifact_path='models')

            # Log preprocessor if exists
            preprocessor_path = self.config['artifacts'].get('preprocessor_path', 'models/preprocessor.pkl')
            if Path(preprocessor_path).exists():
                mlflow.log_artifact(preprocessor_path, artifact_path='preprocessor')

            # Model save via mlflow.sklearn
            mlflow.sklearn.log_model(self.model, artifact_path='sklearn-model')

            logger.info("Training and evaluation completed!")
            
            return self.model, self.metrics


def main():
    """Main function to train the model."""
    import os
    
    # Change to project directory
    project_dir = Path(__file__).resolve().parents[2]
    os.chdir(project_dir)
    
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = CoffeeModelTrainer(config_path='config.yaml')
    
    # Load data
    train_path = config['data']['processed_path'].replace('processed_coffee_sales.csv', 'train_data.csv')
    test_path = config['data']['processed_path'].replace('processed_coffee_sales.csv', 'test_data.csv')
    
    # Check if processed data exists
    processed_dir = Path(config['data']['processed_path']).parent
    train_path = processed_dir / 'train_data.csv'
    test_path = processed_dir / 'test_data.csv'
    
    if not train_path.exists() or not test_path.exists():
        logger.error("Processed data files not found. Please run data preprocessing first.")
        return
    
    X_train, X_test, y_train, y_test = trainer.load_data(str(train_path), str(test_path))
    
    # Train and evaluate
    trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Save model and metrics
    trainer.save_model(config['artifacts']['model_path'])
    trainer.save_metrics(config['artifacts']['metrics_path'])
    
    logger.info("Model training completed!")


if __name__ == '__main__':
    main()
