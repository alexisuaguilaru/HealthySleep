from fastapi import FastAPI
from ModelAPI import InputMLModel , OutputMLModel

app = FastAPI()

@app.post(
    '/Classify',
    summary = 'Method for classify the quality of sleep of a patient',
    response_model = OutputMLModel,
)
def Classify(Input: InputMLModel):
    return {'LevelQualitySleep':0,'NameQualitySleep':'Low'}