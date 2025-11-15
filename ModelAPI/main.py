from fastapi import FastAPI , Depends
from ModelAPI import InputMLModel , OutputMLModel , LoadModel , ClassifyPatient , GetDatabaseConnection , InsertModelInference
from psycopg2.extensions import connection

import pandas as pd

ML_Model = LoadModel('./','ML_Model')

app = FastAPI()

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