from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from seaborn import color_palette
from functools import partial

from matplotlib.figure import Figure
from matplotlib.axes import Axes

BaseColor = '#B57EDC'
BaseComplementColor = '#E1D02A'
BaseTransitionColor = '#FFFFFF'

ColorMapContrast = LinearSegmentedColormap.from_list('CMContrast',[BaseComplementColor,BaseTransitionColor,BaseColor],256)
BasePalette = partial(color_palette,palette='husl')

def CreatePlot(
        NumRows: int = 1,
        NumCols: int = 1,
        FigSize: tuple[float,float] = (6.4,4.8),
    ) -> tuple[Figure,Axes]:
    """
    Function for creating a blank canvas for 
    plotting.

    Parameters
    ----------
    NumRows: int
        Number of rows in the figure

    NumCols: int
        Number of columns in the figure

    FigSize: tuple[float,float]
        Size of the figure

    Returns
    -------
    Fig: Figure
        Figure to show

    Axes: Axes
        Axes for creating plots
    """

    Fig , Axes = plt.subplots(
        NumRows,NumCols,
        figsize = FigSize,
        layout = 'constrained',
        gridspec_kw={'wspace':0.1,'hspace':0.1},
        subplot_kw = {'frame_on':False},
    )

    return Fig , Axes