from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Ładowanie modelu i preprocesora
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Definicja aplikacji FastAPI
app = FastAPI()


# Definicja danych wejściowych
class InputData(BaseModel):
    gender: str
    ethnicity: str
    fcollege: str
    mcollege: str
    home: str
    urban: str
    unemp: float
    wage: float
    distance: float
    tuition: float
    education: int
    income: str
    region: str


# Endpoint przewidywania
@app.post("/predict")
def predict(data: InputData):
    # Przygotowanie danych wejściowych
    input_df = pd.DataFrame([data.dict()])
    input_transformed = preprocessor.transform(input_df)

    # Przewidywanie
    prediction = model.predict(input_transformed)
    return {"predicted_score": prediction[0]}
