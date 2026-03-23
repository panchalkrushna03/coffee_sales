import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CoffeeModelPredictor:
    """
    Predictor class for making predictions with trained coffee sales model.
    Handles model loading, predictions, and evaluation.
    """
    
    def __init__(self, config_path="config.yaml"):
        """Initialize predictor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.preprocessor = None
    
    def load_model(self, model_path):
        """Load trained model from pickle file."""
        logger.info(f"Loading model from {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        logger.info("Model loaded successfully!")
        return self.model
    
    def load_preprocessor(self, preprocessor_path):
        """Load preprocessor from pickle file."""
        logger.info(f"Loading preprocessor from {preprocessor_path}")
        
        if not Path(preprocessor_path).exists():
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
        
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        logger.info("Preprocessor loaded successfully!")
        return self.preprocessor
    
    def predict(self, X):
        """Make predictions on new data."""
        logger.info(f"Making predictions on {X.shape[0]} samples")
        
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model first.")
        
        predictions = self.model.predict(X)
        
        logger.info(f"Predictions completed! Shape: {predictions.shape}")
        return predictions
    
    def evaluate_on_test_data(self, X_test, y_test):
        """Evaluate model on test data."""
        logger.info("Evaluating model on test data...")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Flatten y_test if needed
        y_test = np.ravel(y_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        metrics = {
            'r2_score': float(r2),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae)
        }
        
        logger.info("=" * 50)
        logger.info("TEST DATA EVALUATION METRICS")
        logger.info("=" * 50)
        logger.info(f"R² Score: {r2:.4f}")
        logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")
        logger.info("=" * 50)
        
        return metrics
    
    def predict_single(self, sample):
        """Make prediction on a single sample."""
        logger.info("Making prediction on single sample")
        
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model first.")
        
        # Reshape sample if needed
        if len(sample.shape) == 1:
            sample = sample.reshape(1, -1)
        
        prediction = self.model.predict(sample)
        
        logger.info(f"Prediction: {prediction[0]:.4f}")
        return prediction[0]


def main():
    """Main function to demonstrate predictions."""
    import os
    
    # Change to project directory
    project_dir = Path(__file__).resolve().parents[2]
    os.chdir(project_dir)
    
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize predictor
    predictor = CoffeeModelPredictor(config_path='config.yaml')
    
    # Load model and preprocessor
    try:
        predictor.load_model(config['artifacts']['model_path'])
        predictor.load_preprocessor(config['artifacts']['preprocessor_path'])
    except FileNotFoundError as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Please train the model first using train_model.py")
        return
    
    # Load test data for evaluation
    processed_dir = Path(config['data']['processed_path']).parent
    test_path = processed_dir / 'test_data.csv'
    
    if test_path.exists():
        logger.info(f"Loading test data from {test_path}")
        test_data = pd.read_csv(test_path)
        
        # Separate features and target
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        
        # Evaluate on test data
        metrics = predictor.evaluate_on_test_data(X_test, y_test)
        
        logger.info("Evaluation completed successfully!")
    else:
        logger.warning("Test data not found. Skipping evaluation.")
    
    logger.info("Prediction module ready!")


if __name__ == '__main__':
    main()
