from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
import yaml
from datetime import datetime

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Coffee Sales Price Prediction API",
    description="API for predicting coffee prices based on various features",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
preprocessor = None
config = None
metrics = None

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    store_id: int = Field(..., description="Store ID")
    city: str = Field(..., description="City name")
    country: str = Field(..., description="Country code")
    store_type: str = Field(..., description="Type of store (e.g., Standalone, Mall Kiosk)")
    product_category: str = Field(..., description="Product category (e.g., Coffee, Tea)")
    product_name: str = Field(..., description="Product name")
    quantity: int = Field(..., description="Quantity ordered")
    discount_applied: bool = Field(..., description="Whether discount was applied")
    payment_method: str = Field(..., description="Payment method")
    customer_age_group: str = Field(..., description="Customer age group")
    customer_gender: str = Field(..., description="Customer gender (Male/Female)")
    loyalty_member: bool = Field(..., description="Whether customer is loyalty member")
    weather_condition: str = Field(..., description="Weather condition")
    temperature_c: float = Field(..., description="Temperature in Celsius")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    is_weekend: int = Field(..., ge=0, le=1, description="Is weekend (0/1)")

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    records: List[PredictionRequest] = Field(..., description="List of prediction requests")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_price: float = Field(..., description="Predicted coffee price")
    confidence: str = Field(default="High", description="Prediction confidence level")
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    count: int = Field(..., description="Number of predictions")

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_type: str = Field(..., description="Type of model")
    r2_score: float = Field(..., description="R² Score")
    mse: float = Field(..., description="Mean Squared Error")
    rmse: float = Field(..., description="Root Mean Squared Error")
    mae: float = Field(..., description="Mean Absolute Error")
    created_date: str = Field(..., description="Model creation date")
    features_count: int = Field(..., description="Number of input features")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    preprocessor_loaded: bool = Field(..., description="Whether preprocessor is loaded")
    timestamp: str = Field(..., description="Check timestamp")


def load_model_and_preprocessor():
    """Load trained model and preprocessor from pickle files"""
    global model, preprocessor, config, metrics
    
    logger.info("Loading model and preprocessor...")
    
    try:
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Load model
        model_path = Path(config['artifacts']['model_path'])
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}")
        
        # Load preprocessor
        preprocessor_path = Path(config['artifacts']['preprocessor_path'])
        if preprocessor_path.exists():
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        else:
            logger.warning(f"Preprocessor file not found at {preprocessor_path}")
        
        # Load metrics
        metrics_path = Path(config['artifacts']['metrics_path'])
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            logger.info(f"Metrics loaded from {metrics_path}")
        else:
            logger.warning(f"Metrics file not found at {metrics_path}")
            metrics = {}
        
        logger.info("Model and preprocessor loaded successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False


def prepare_features(data: Dict[str, Any]) -> np.ndarray:
    """
    Prepare features for model prediction
    """
    try:
        # Create a DataFrame from the input data
        df = pd.DataFrame([data])
        
        # Define feature order based on the model's expectations
        numeric_features = [
            'store_id', 'quantity', 'discount_applied', 'temperature_c',
            'month', 'day_of_week', 'hour', 'is_weekend'
        ]
        
        categorical_features = [
            'city', 'country', 'store_type', 'product_category', 'product_name',
            'payment_method', 'customer_age_group', 'customer_gender',
            'loyalty_member', 'weather_condition'
        ]
        
        # Ensure numeric features are numeric
        for col in numeric_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Select only the features needed
        feature_cols = numeric_features + categorical_features
        X = df[feature_cols]
        
        # Transform using preprocessor
        X_transformed = preprocessor.transform(X)
        
        return X_transformed
    
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model and preprocessor on startup"""
    logger.info("Starting up FastAPI application...")
    success = load_model_and_preprocessor()
    
    if not success:
        logger.warning("Failed to load model and preprocessor on startup")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Coffee Sales Price Prediction API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "model_info": "/model-info"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API and model health status"""
    return HealthResponse(
        status="healthy" if model and preprocessor else "unhealthy",
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get model information and evaluation metrics"""
    if not model or not metrics or not config:
        raise HTTPException(
            status_code=503,
            detail="Model or metrics not loaded"
        )
    
    return ModelInfoResponse(
        model_type=config['model']['type'],
        r2_score=metrics.get('r2_score', 0.0),
        mse=metrics.get('mse', 0.0),
        rmse=metrics.get('rmse', 0.0),
        mae=metrics.get('mae', 0.0),
        created_date=datetime.now().isoformat(),
        features_count=model.n_features_in_ if hasattr(model, 'n_features_in_') else 0
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a single coffee price prediction
    
    Example request:
    ```json
    {
        "store_id": 1,
        "city": "Los Angeles",
        "country": "USA",
        "store_type": "Standalone",
        "product_category": "Coffee",
        "product_name": "Large Cappuccino",
        "quantity": 1,
        "discount_applied": false,
        "payment_method": "Credit Card",
        "customer_age_group": "25-34",
        "customer_gender": "Male",
        "loyalty_member": true,
        "weather_condition": "Sunny",
        "temperature_c": 15.5,
        "month": 3,
        "day_of_week": 2,
        "hour": 10,
        "is_weekend": 0
    }
    ```
    """
    if not model or not preprocessor:
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor not loaded"
        )
    
    try:
        # Prepare features
        X = prepare_features(request.dict())
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        return PredictionResponse(
            predicted_price=float(prediction),
            confidence="High" if metrics.get('r2_score', 0) > 0.8 else "Medium",
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error making prediction: {str(e)}"
        )


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions for multiple records
    """
    if not model or not preprocessor:
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor not loaded"
        )
    
    try:
        predictions = []
        
        for item in request.records:
            try:
                # Prepare features
                X = prepare_features(item.dict())
                
                # Make prediction
                prediction = model.predict(X)[0]
                
                predictions.append(
                    PredictionResponse(
                        predicted_price=float(prediction),
                        confidence="High" if metrics.get('r2_score', 0) > 0.8 else "Medium",
                        timestamp=datetime.now().isoformat()
                    )
                )
            except Exception as e:
                logger.error(f"Error predicting for record: {str(e)}")
                raise
        
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error making batch predictions: {str(e)}"
        )


@app.post("/predict-sample", response_model=PredictionResponse, tags=["Prediction"])
async def predict_sample():
    """
    Get a sample prediction with default values for testing
    """
    if not model or not preprocessor:
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor not loaded"
        )
    
    sample_request = PredictionRequest(
        store_id=1,
        city="Los Angeles",
        country="USA",
        store_type="Standalone",
        product_category="Coffee",
        product_name="Large Cappuccino",
        quantity=1,
        discount_applied=False,
        payment_method="Credit Card",
        customer_age_group="25-34",
        customer_gender="Male",
        loyalty_member=True,
        weather_condition="Sunny",
        temperature_c=15.5,
        month=3,
        day_of_week=2,
        hour=10,
        is_weekend=0
    )
    
    return await predict(sample_request)


@app.get("/model-features", tags=["Model"])
async def get_model_features():
    """Get information about model features"""
    if not model:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "total_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else 0,
        "numeric_features": [
            'store_id', 'quantity', 'discount_applied', 'temperature_c',
            'month', 'day_of_week', 'hour', 'is_weekend'
        ],
        "categorical_features": [
            'city', 'country', 'store_type', 'product_category', 'product_name',
            'payment_method', 'customer_age_group', 'customer_gender',
            'loyalty_member', 'weather_condition'
        ]
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Coffee Sales Price Prediction API...")
    logger.info("Access the API at http://localhost:8000")
    logger.info("Access the interactive docs at http://localhost:8000/docs")
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
