from pickle import load as ml_load
import pandas as pd

from sklearn.pipeline import Pipeline
from .Models import *

def LoadModel(
        PathDir: str,
        NameModel: str,
    ) -> Pipeline:
    """
    Function for loading a 
    model/pipeline of scikit-learn
    from a dir with a given name.

    Parameters
    ----------
    PathDir: str
        Path from where is loaded the model
    NameModel: str
        Name of the model to load

    Return
    ------
    Model: Pipeline
        Loaded model/pipeline
    """

    with open(f'{PathDir}{NameModel.replace(' ','')}.pkl','rb') as file_model:
        model = ml_load(file_model)
        return model

def ClassifyPatient(
        InputPatient: InputMLModel,
        ML_Model: Pipeline,
    ) -> OutputMLModel:
    """
    Function for classify the quality of sleep of a patient.

    Parameters
    ----------
    InputPatient: InputMLModel
        Features (Values) of a patient.
    ML_Model: Pipeline
        Loaded model/pipeline

    Return
    ------
    OutputMLModel: OutputMLModel
        Predicted quality of sleep for a patient.
    """

    FeatureValues = InputPatient.model_dump(by_alias=True)
    Input = pd.DataFrame([FeatureValues])

    LevelQualitySleep = ML_Model.predict(Input)[0]
    return OutputMLModel(LevelQualitySleep=LevelQualitySleep)