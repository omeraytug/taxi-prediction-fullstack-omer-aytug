from pathlib import Path

DATA_PATH = Path(__file__).parents[1] / "data"
TAXI_CSV_PATH = DATA_PATH / "taxi_trip_pricing.csv"
NEW_CSV_PATH = DATA_PATH / "taxi_trip_pricing_cleaned.csv"

MODEL_PATH = Path(__file__).parents[1] / "models"
MODEL = MODEL_PATH / "random_forest_model.joblib"