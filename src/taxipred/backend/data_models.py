from pydantic import BaseModel, Field
from typing import Optional, Literal
from taxipred.utils.constants import NEW_CSV_PATH
import pandas as pd
import json

class TaxiData:
    def __init__(self):
        self.df = pd.read_csv(NEW_CSV_PATH)

    def to_json(self):
        return json.loads(self.df.to_json(orient="records"))
    
class TaxiInput(BaseModel):
    """Input model for taxi price prediction.
    
    Accepts raw features that will be preprocessed (dummy encoded) 
    to match the model's expected input format.
    """
    trip_distance_km: float = Field(gt=0, description="Trip distance in kilometers")
    trip_duration_minutes: float = Field(gt=0, description="Trip duration in minutes")
    time_of_day: Literal["Afternoon", "Evening", "Morning", "Night"] = Field(
        default="Evening", 
        description="Time of day category"
    )
    day_of_week: Literal["Weekday", "Weekend"] = Field(
        default="Weekend",
        description="Day of week category"
    )
    traffic_conditions: Literal["High", "Low", "Medium"] = Field(
        default="Low",
        description="Traffic conditions"
    )
    weather: Literal["Clear", "Rain", "Snow"] = Field(
        default="Clear",
        description="Weather conditions"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "trip_distance_km": 15.5,
                "trip_duration_minutes": 30.0,
                "time_of_day": "Evening",
                "day_of_week": "Weekend",
                "traffic_conditions": "Low",
                "weather": "Clear"
            }
        }

class PredictionResponse(BaseModel):
    """Response model for taxi price prediction."""
    predicted_price: float = Field(description="Predicted trip price")
    input_features: dict = Field(description="Processed input features used for prediction")
    