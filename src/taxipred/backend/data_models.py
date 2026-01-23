from pydantic import BaseModel, Field
from typing import Optional
from taxipred.utils.constants import NEW_CSV_PATH
import pandas as pd
import json

class TaxiData:
    def __init__(self):
        self.df = pd.read_csv(NEW_CSV_PATH)

    def to_json(self):
        return json.loads(self.df.to_json(orient="records"))
    
class TaxiInput(BaseModel):
    trip_distance_km: float
    trip_duration_minutes: Optional[float] = Field(default=None, ge=0)
