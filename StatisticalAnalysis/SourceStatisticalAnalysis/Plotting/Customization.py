from matplotlib.axes import Axes
from matplotlib.figure import Figure

def SetLabelsToPlot(
        Ax: Axes,
        Tittle: str = None,
        LabelX: str = None,
        LabelY: str = None,
        TitleSize: int = 10,
        LabelSize: int = 8,
        TickSize: int = 7,
    ) -> None:
    """
    Function for setting the label 
    and their sizes of a plot

    Parameters
    ----------
    Ax: Axes
        Plot to set its size

    Title: str = None
        Plot tittle

    LabelX: str = None
        Label of axis X

    LabelY: str = None
        Label of axis Y

    TitleSize: int = 14
        Size of tittle

    LabelSize: int = 10
        Size of the labels
    
    TickSize: int
        Size of the ticklabels
    """
    
    if Tittle: Ax.set_title(Tittle,size=TitleSize)
    if LabelX: Ax.set_xlabel(LabelX,size=LabelSize)
    if LabelY: Ax.set_ylabel(LabelY,size=LabelSize)
    Ax.tick_params(axis='both',labelsize=TickSize)

def SetFigureTitle(
        Fig: Figure,
        Tittle: str = None,
        Size: int = 14,
    ) -> None:
    """
    Function for setting the tittle of 
    the figure.

    Parameters
    ----------
    Fig: Figure
        Figure to set its tittle

    Tittle: str
        Tittle of the figure

    Size: int
        Size of the tittle
    """

    Fig.suptitle(Tittle,size=Size)