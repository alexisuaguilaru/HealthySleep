import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def CategorizeBloodPressure(
        Systolic: int,
        Diastolic: int,
    ) -> str:
    """
    Function for categorizing the blood 
    pressure of a patient based on their 
    systolic and diastolic pressure.

    Parameter
    ---------
    Systolic: int
        Systolic pressure measure

    Diastolic: int
        Diastolic pressure measure

    Returns
    -------
    EncodedBloodPressure: str
        Category of the blood pressure of a patient
    """

    EncodedBloodPressure = None
    if Systolic < 120 and Diastolic < 80:
        EncodedBloodPressure = 'Normal'
    elif Systolic < 130 and Diastolic < 80:
        EncodedBloodPressure = 'Elevated'
    elif Systolic < 140 and Diastolic < 90:
        EncodedBloodPressure = 'Hypertension Stage 1'
    elif Systolic < 180 and Diastolic < 120:
        EncodedBloodPressure = 'Hypertension Stage 2'
    else:
        EncodedBloodPressure = 'Hypertensive Crisis'
    
    return EncodedBloodPressure

def OneHotEncoderFeature( 
        SleepDataset: pd.DataFrame,
        Feature: str,
        BinsRange: list[int],
        LabelsRange: list[str],
    ) -> None:
    """
    """

    if BinsRange and LabelsRange:
        EncodedValuesFeature = pd.cut(
            SleepDataset[Feature],
            BinsRange,
            labels = LabelsRange,
            ordered = False,
        )
        EncodedValuesFeature = np.reshape(EncodedValuesFeature,(-1,1))
    else:
        EncodedValuesFeature = SleepDataset[[Feature]]

    SleepDataset.drop(columns=Feature,inplace=True)
    TransformerOHE = OneHotEncoder(sparse_output=False)
    OneHotValues = TransformerOHE.fit_transform(EncodedValuesFeature)
    SleepDataset[Feature + ' :: ' + np.array(*TransformerOHE.categories_,dtype=str)] = OneHotValues