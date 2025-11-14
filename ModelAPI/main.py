from fastapi import FastAPI
from ModelAPI import InputMLModel , OutputMLModel , LoadModel , ClassifyPatient

import pandas as pd

ML_Model = LoadModel('./','ML_Model')

app = FastAPI()

@app.post(
    '/Classify',
    summary = 'Method for classify the quality of sleep of a patient',
    response_model = OutputMLModel,
)
def Classify(InputPatient: InputMLModel):
    return ClassifyPatient(InputPatient,ML_Model)