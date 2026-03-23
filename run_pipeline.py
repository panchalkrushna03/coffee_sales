#!/usr/bin/env python
"""
Main script to run the complete coffee sales price prediction pipeline.
This script orchestrates data preprocessing, model training, and evaluation.
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import yaml

# Add src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from src.preprocessing import CoffeeDataPreprocessor
from src.models.train_model import CoffeeModelTrainer
from src.models.predict_model import CoffeeModelPredictor

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def create_directories(config):
    """Create necessary directories for the project."""
    logger.info("Creating project directories...")
    
    dirs = [
        Path(config['data']['processed_path']).parent,
        Path(config['data']['interim_path']),
        Path(config['artifacts']['model_path']).parent,
        Path(config['artifacts']['metrics_path']).parent,
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/confirmed directory: {dir_path}")


def run_preprocessing(config):
    """Run data preprocessing pipeline."""
    logger.info("=" * 70)
    logger.info("STARTING DATA PREPROCESSING")
    logger.info("=" * 70)
    
    # Initialize preprocessor
    preprocessor = CoffeeDataPreprocessor(config_path='config.yaml')
    
    # Preprocess data
    X_processed, y, feature_names = preprocessor.preprocess(
        config['data']['raw_path']
    )
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    logger.info('Splitting data into train and test sets')
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, 
        y,
        test_size=config['preprocessing']['test_size'],
        random_state=config['preprocessing']['random_state']
    )
    
    # Save preprocessor
    logger.info('Saving preprocessor...')
    preprocessor.save_preprocessor(config['artifacts']['preprocessor_path'])
    
    # Create processed data directory
    processed_dir = Path(config['data']['processed_path']).parent
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train and test data
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    y_train_df = pd.DataFrame(y_train, columns=[config['preprocessing']['target_variable']])
    y_test_df = pd.DataFrame(y_test, columns=[config['preprocessing']['target_variable']])
    
    train_data = pd.concat([X_train_df.reset_index(drop=True), y_train_df.reset_index(drop=True)], axis=1)
    test_data = pd.concat([X_test_df.reset_index(drop=True), y_test_df.reset_index(drop=True)], axis=1)
    
    train_path = processed_dir / 'train_data.csv'
    test_path = processed_dir / 'test_data.csv'
    
    logger.info(f'Saving processed train data to {train_path}')
    train_data.to_csv(train_path, index=False)
    
    logger.info(f'Saving processed test data to {test_path}')
    test_data.to_csv(test_path, index=False)
    
    logger.info("Data preprocessing completed successfully!")
    logger.info(f'Train set shape: {train_data.shape}')
    logger.info(f'Test set shape: {test_data.shape}')
    logger.info("=" * 70)
    
    return train_path, test_path


def run_training(config, train_path, test_path):
    """Run model training pipeline."""
    logger.info("=" * 70)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 70)
    
    # Initialize trainer
    trainer = CoffeeModelTrainer(config_path='config.yaml')
    
    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data(str(train_path), str(test_path))
    
    # Train and evaluate
    trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Save model and metrics
    Path(config['artifacts']['model_path']).parent.mkdir(parents=True, exist_ok=True)
    Path(config['artifacts']['metrics_path']).parent.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(config['artifacts']['model_path'])
    trainer.save_metrics(config['artifacts']['metrics_path'])
    
    logger.info("=" * 70)
    logger.info("Model training completed successfully!")
    logger.info("=" * 70)


def run_evaluation(config):
    """Run model evaluation pipeline."""
    logger.info("=" * 70)
    logger.info("STARTING MODEL EVALUATION")
    logger.info("=" * 70)
    
    # Initialize predictor
    predictor = CoffeeModelPredictor(config_path='config.yaml')
    
    # Load model and preprocessor
    try:
        predictor.load_model(config['artifacts']['model_path'])
        predictor.load_preprocessor(config['artifacts']['preprocessor_path'])
    except FileNotFoundError as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Load test data
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
    
    logger.info("=" * 70)
    logger.info("Model evaluation completed!")
    logger.info("=" * 70)


def main():
    """Main function to run the complete pipeline."""
    logger.info("\n")
    logger.info("=" * 70)
    logger.info("COFFEE SALES PRICE PREDICTION - COMPLETE PIPELINE")
    logger.info("=" * 70)
    
    # Load configuration
    config_path = Path('config.yaml')
    if not config_path.exists():
        logger.error("config.yaml not found in current directory!")
        return
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from: {config_path.absolute()}")
    
    # Create directories
    create_directories(config)
    
    # Run preprocessing
    train_path, test_path = run_preprocessing(config)
    
    # Run training
    run_training(config, train_path, test_path)
    
    # Run evaluation
    run_evaluation(config)
    
    logger.info("\n")
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    logger.info("\nGenerated Artifacts:")
    logger.info(f"  - Model: {config['artifacts']['model_path']}")
    logger.info(f"  - Preprocessor: {config['artifacts']['preprocessor_path']}")
    logger.info(f"  - Metrics: {config['artifacts']['metrics_path']}")
    logger.info(f"  - Train Data: {Path(config['data']['processed_path']).parent / 'train_data.csv'}")
    logger.info(f"  - Test Data: {Path(config['data']['processed_path']).parent / 'test_data.csv'}")
    logger.info("=" * 70 + "\n")


if __name__ == '__main__':
    main()
