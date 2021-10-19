from io import StringIO

import joblib
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from src.animal_shelter.model import predict_model, load_model
from src.animal_shelter.data import load_data

import pandas as pd


app = FastAPI()

@app.get("/api/v1/ping")
def ping():
    return "pong"


@app.post("/api/v1/predict/")
def predict(file: UploadFile = File(...)):
    input_data = load_data(file.file)

    model = load_model('output/trained_model.pkl')
    predictions = predict_model(model, input_data)
    response = _convert_df_to_response(predictions)
    return response


def _convert_df_to_response(df: pd.DataFrame) -> StreamingResponse:
    """Convert a DataFrame to CSV response."""
    stream = StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(
        iter([stream.getvalue()]), media_type="text/csv"
    )
    return response
