import seaborn as sns
from ..Plotting import CreatePlot , SetLabelsToPlot , BaseColor , BaseComplementColor
from matplotlib.figure import Figure
import numpy as np

def PlotObservedPredictedValues(
        ObservedValues: np.ndarray,
        PredictedValues: np.ndarray,
        ModelType: str,
    ) -> Figure:
    """
    Function for plotting the predicted values of 
    linear regression versus observed (true) values.

    Parameters
    ----------
    ObservedValues: np.ndarray
        True target values 

    PredictedValues: np.ndarray
        Predicted target values

    ModelType: str
        Model's type (whether is full, best or reduced)

    Return
    ------
    Fig: Figure
        Figure with the plot of observations vs. predictions
    """

    Fig , Axes = CreatePlot(FigSize=(5,5))

    sns.scatterplot(
        x = ObservedValues,
        y = PredictedValues,
        color = BaseColor,
        ax = Axes,
    )

    Axes.axline((1, 1),slope=1,color=BaseComplementColor,linestyle='--',)

    SetLabelsToPlot(
        Axes,
        f'Quality of {ModelType} Linear Regression Model',
        'Observed Values',
        'Predicted Values',
        13,11,9
    )

    return Fig