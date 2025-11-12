import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import pandas as pd
from matplotlib.figure import Figure

BaseColor = '#B57EDC'
BaseTransitionColor = '#FFFFFF'
BaseComplementColor = '#E1D02A'

ColorMapBinary = LinearSegmentedColormap.from_list('CMContrast',[BaseTransitionColor,BaseColor],256)

def PlotResults(
        Results: pd.DataFrame,
        TypeDataset: str,    
    ):
    """
    Function for plot the results of 
    model evaluations.

    Parameters
    ----------
    Results: pd.DataFrame
        `pd.DataFrame` with the scores of each model
    TypeDataset: str
        Type of dataset used for getting the results

    Return
    ------
    Plot: Figure
        Plot with the results
    """

    Fig , Axes = plt.subplots(
        subplot_kw={'frame_on':False,'ylim':(0.9,1)},
    )

    Results.plot(
        kind='bar',
        ax=Axes,
        legend=False,
        color=sns.color_palette('husl'),
    )
    Axes.grid(True,axis='y',color='gray',lw=1,ls=':')

    Axes.set_yticks(np.linspace(0.9,1,11))
    TicksLabels = Axes.get_xticklabels()
    Axes.set_xticks(range(len(TicksLabels)),TicksLabels,rotation=30)

    Axes.tick_params(axis='both',labelsize=10,width=0)
    Axes.set_xlabel(Axes.get_xlabel(),size=12)
    Axes.set_ylabel('Score',size=12)
    Axes.set_title(f'Results on\n{TypeDataset} Dataset',size=15)

    Axes.legend(fontsize=11)

    return Fig