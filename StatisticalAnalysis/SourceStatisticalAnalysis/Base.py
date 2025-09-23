from matplotlib.colors import LinearSegmentedColormap

BaseColor = '#B57EDC'
BaseComplementColor = '#E1D02A'
BaseTransitionColor = '#FFFFFF'
ColorMapContrast = LinearSegmentedColormap.from_list('CMContrast',[BaseComplementColor,BaseTransitionColor,BaseColor],256)