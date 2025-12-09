import seaborn as sns
from ..Plotting import CreatePlot , SetLabelsToPlot , BaseColor
from matplotlib.figure import Figure

def PlotSilhouetteResults(
        ValuesCriterion: list[int|float],
        SilhouetteScores: list[float],
        CriterionName: str,
        ClusteringTechnique: str,
    ) -> Figure:
    """
    Function for plotting the results of Silhouette 
    score over a list of parameters/clustering models.

    Parameters
    ----------
    ValuesCriterion: list[int|float]
        List of values for generating each model

    SilhouetteScores: list[float]
        Silhouette score for each model

    CriterionName: str
        Name of the parameter being varied

    ClusteringTechnique: str
        Clustering algorithm name


    Return
    ------
    Fig: Figure
        Figure with the plot of the results
    """

    Fig , Axes = CreatePlot(FigSize=(6,5))
    
    sns.lineplot(
        x = ValuesCriterion,
        y = SilhouetteScores,
        color = BaseColor,
        linestyle = '--',
        linewidth = 1.5,
        marker = 'o',
        markersize = 6,
        ax = Axes,
    )
    SetLabelsToPlot(
        Axes,
        f'Scree Plot for Selection of\n{CriterionName} in {ClusteringTechnique}',
        CriterionName,
        'Silhouette Score',
        13,11,10
    )

    return Fig