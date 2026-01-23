from fastapi import FastAPI
from taxipred.backend.data_models import TaxiData, TaxiInput, PredictionResponse
from taxipred.utils.constants import MODEL
import joblib
import pandas as pd

app = FastAPI(
    title="Taxi Price Prediction API",
    description="API for predicting taxi trip prices using machine learning",
    version="1.0.0"
)

# Load the trained model
model = joblib.load(MODEL)

taxi_data = TaxiData()

# Expected feature order from the trained model
# Based on the training notebook: ['trip_distance_km', 'trip_duration_minutes', 
# 'time_of_day_Evening', 'time_of_day_Morning', 'time_of_day_Night', 
# 'day_of_week_Weekend', 'traffic_conditions_Low', 'traffic_conditions_Medium', 
# 'weather_Rain', 'weather_Snow']
EXPECTED_FEATURES = [
    'trip_distance_km',
    'trip_duration_minutes',
    'time_of_day_Evening',
    'time_of_day_Morning',
    'time_of_day_Night',
    'day_of_week_Weekend',
    'traffic_conditions_Low',
    'traffic_conditions_Medium',
    'weather_Rain',
    'weather_Snow'
]


def preprocess_input(input_data: TaxiInput) -> pd.DataFrame:
    """
    Preprocess input data to match the model's expected format.
    Performs dummy encoding similar to the training pipeline.
    """
    # Create a DataFrame with the raw input
    df = pd.DataFrame([{
        'trip_distance_km': input_data.trip_distance_km,
        'trip_duration_minutes': input_data.trip_duration_minutes,
        'time_of_day': input_data.time_of_day,
        'day_of_week': input_data.day_of_week,
        'traffic_conditions': input_data.traffic_conditions,
        'weather': input_data.weather
    }])
    
    # Separate numerical and categorical columns
    numeric_columns = ['trip_distance_km', 'trip_duration_minutes']
    cat_columns = ['time_of_day', 'day_of_week', 'traffic_conditions', 'weather']
    
    # Extract numerical features
    X_num = df[numeric_columns].copy()
    
    # Create dummy variables for categorical features (drop_first=True as in training)
    X_cat = df[cat_columns].copy()
    X_cat_dummies = pd.get_dummies(X_cat, columns=cat_columns, drop_first=True)
    
    # Combine numerical and dummy-encoded categorical features
    X_processed = pd.concat([X_num, X_cat_dummies], axis=1)
    
    # Ensure all expected features are present, fill missing with 0
    for feature in EXPECTED_FEATURES:
        if feature not in X_processed.columns:
            X_processed[feature] = 0
    
    # Reorder columns to match expected feature order
    X_processed = X_processed[EXPECTED_FEATURES]
    
    return X_processed


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Taxi Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /taxi/": "Get all taxi data",
            "POST /predict/": "Predict taxi price"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint to verify API and model status."""
    return {
        "status": "healthy",
        "model_path": str(MODEL)
    }


@app.get("/taxi/")
async def read_taxi_data():
    """Get all taxi data from the cleaned dataset."""
    return taxi_data.to_json()


@app.post("/predict/", response_model=PredictionResponse)
async def predict_price(input_data: TaxiInput):
    """
    Predict taxi trip price based on input features.
    
    The input features are preprocessed (dummy encoded) to match
    the format expected by the trained model.
    """
    # Preprocess input to match model format
    X_processed = preprocess_input(input_data)
    
    # Make prediction
    prediction = model.predict(X_processed)[0]
    
    # Prepare response
    return PredictionResponse(
        predicted_price=float(prediction),
        input_features=X_processed.iloc[0].to_dict()
    )

