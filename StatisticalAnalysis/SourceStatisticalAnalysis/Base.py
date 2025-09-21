from matplotlib.colors import LinearSegmentedColormap

BaseColor = '#B57EDC'
BaseComplementColor = '#E1D02A'
ColorMapContrast = LinearSegmentedColormap.from_list('CMContrast',[BaseComplementColor,BaseColor],256)