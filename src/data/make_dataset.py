# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
import pickle
import yaml

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing import CoffeeDataPreprocessor


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize preprocessor
    preprocessor = CoffeeDataPreprocessor(config_path='config.yaml')
    
    # Preprocess data
    logger.info('Preprocessing data...')
    X_processed, y, feature_names = preprocessor.preprocess(input_filepath)
    
    # Split data
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
    
    # Create DataFrames for easier handling
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)
    
    # Save processed data
    logger.info(f'Saving processed data to {output_filepath}')
    
    # Save train data
    train_data = pd.concat([X_train_df, y_train_df.reset_index(drop=True)], axis=1)
    train_data.to_csv(f"{output_filepath}/train_data.csv", index=False)
    
    # Save test data
    test_data = pd.concat([X_test_df, y_test_df.reset_index(drop=True)], axis=1)
    test_data.to_csv(f"{output_filepath}/test_data.csv", index=False)
    
    # Save feature names and schema info
    with open(f"{output_filepath}/feature_names.pkl", 'wb') as f:
        pickle.dump(feature_names, f)
    
    logger.info('Data processing completed successfully!')
    logger.info(f'Train set shape: {train_data.shape}')
    logger.info(f'Test set shape: {test_data.shape}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Get project directory
    project_dir = Path(__file__).resolve().parents[2]
    
    # Change to project directory
    import os
    os.chdir(project_dir)

    # Load environment variables
    load_dotenv(find_dotenv())

    main()
