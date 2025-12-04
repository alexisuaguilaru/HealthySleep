from sklearn.linear_model import LinearRegression
import numpy as np

from typing import Any

def AkaikeInformationCriterionScore(
        LinearModel: LinearRegression,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
    """
    Function for calculating the AIC 
    score of a linear regression model 
    using a evaluation dataset.

    Parameters
    ----------
    LinearModel: LinearRegression
        Fitted linear model regression to evaluate

    X: np.ndarray
        X evaluation

    y: np.ndarray
        y evaluation

    Return
    ------
    aic_score: float
        AIC score of the linear model
    """

    LenDataset = len(y)
    y_pred = LinearModel.predict(X)
    SumSquareErrors = np.mean((y-y_pred)**2)
    DegreeFreedom = X.shape[1] + 1

    return -(LenDataset*np.log(2*np.pi*SumSquareErrors)+LenDataset+2*DegreeFreedom)

def OutputFormatting(
        Value: Any,
        Precision: int = 4
    ) -> str | Any:
    """
    Function for formatting numerical values 
    with 4-digits precision.

    Parameter
    ---------
    Value: Any
        Value to format

    Return
    ------
    FormattedValue: str | Any
        Formatted string
    """
    
    try:
        return round(Value,Precision)
    except:
        return Value