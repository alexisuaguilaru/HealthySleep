import pandas as pd

def SplitFeatures(
        Dataset: pd.DataFrame,
    ) -> tuple[list[str],list[str]]:
    """
    Function for splitting features into 
    numerical and categorical based on 
    their datatypes

    Parameter
    ---------
    Dataset: pd.DataFrame
        Dataset which features are split based on datatypes

    Returns
    -------
    NumericalFeatures: list[str]
        Set of numerical features
    CategoricalFeatures: list[str]
        Set of categorical features
    """

    NumericalFeatures , CategoricalFeatures = [] , []
    for feature in Dataset.columns:
        if Dataset[feature].dtype == 'object':
            CategoricalFeatures.append(feature)
        else:
            NumericalFeatures.append(feature)

    return NumericalFeatures , CategoricalFeatures 

def SplitBloodPressure(
        BloodPressure: str,
    ) -> list[int,int]:
    """
    Function for splitting blood pressure 
    measure into systolic and diastolic 
    pressure.

    Parameter
    ---------
    BloodPressure: str
        Blood pressure of a patient in 
        Systolic/Diastolic form

    Returns
    -------
    Systolic: int
        Systolic pressure measure

    Diastolic: int
        Diastolic pressure measure
    """

    return list(map(int,BloodPressure.split('/')))