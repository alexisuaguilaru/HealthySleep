from matplotlib.colors import LinearSegmentedColormap
from seaborn import color_palette
from functools import partial

BaseColor = '#B57EDC'
BaseComplementColor = '#E1D02A'
BaseTransitionColor = '#FFFFFF'

ColorMapContrast = LinearSegmentedColormap.from_list('CMContrast',[BaseComplementColor,BaseTransitionColor,BaseColor],256)
BasePalette = partial(color_palette,palette='husl')