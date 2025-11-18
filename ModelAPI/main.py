from fastapi import FastAPI , Depends
from fastapi.middleware.cors import CORSMiddleware
from ModelAPI import InputMLModel , OutputMLModel , LoadModel , ClassifyPatient , GetDatabaseConnection , InsertModelInference
from psycopg2.extensions import connection

import pandas as pd

ML_Model = LoadModel('./','ML_Model')

app = FastAPI()
origins = [
    'http://localhost',
    'http://127.0.0.1',
    'http://localhost:5173',
    'http://127.0.0.1:5173',
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['POST','OPTIONS'],
    allow_headers=['Application-Type','Content-Type','Authorization'],
)

@app.post(
    '/Classify',
    summary = 'Method for classify the quality of sleep of a patient',
    response_model = OutputMLModel,
)
def Classify(
        InputPatient: InputMLModel,
        DatabaseConnection: connection = Depends(GetDatabaseConnection),
    ) -> OutputMLModel:
    """
    Method for classify the quality of sleep of a patient.

    Parameters
    ----------
    InputPatient: InputMLModel
        Features (Values) of a patient.
        
    DatabaseConnection: connection
        Database connection.

    Return
    ------
    OutputMLModel: OutputMLModel
        Predicted quality of sleep for a patient.
    """

    ModelInference = ClassifyPatient(InputPatient,ML_Model)
    InsertModelInference(InputPatient,ModelInference,DatabaseConnection)
    return ModelInference