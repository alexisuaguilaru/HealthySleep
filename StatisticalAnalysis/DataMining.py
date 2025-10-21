import marimo

__generated_with = "0.16.5"
app = marimo.App()

with app.setup:
    # Import auxiliar libraries
    import marimo as mo
    from functools import partial
    from itertools import combinations


    # Importing libraries
    import pandas as pd
    import numpy as np

    import seaborn as sns
    import matplotlib.pyplot as plt

    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    from scipy import stats

    from sklearn.decomposition import PCA
    from sklearn.cluster import AgglomerativeClustering , KMeans , DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    from sklearn.metrics import silhouette_score


    # Importing Functions and Utils
    import SourceStatisticalAnalysis as src


@app.cell
def _():
    mo.md(r"##")
    return


@app.cell
def _():
    # Setting constants

    RANDOM_STATE = 8013
    PATH = './Datasets/{}SleepDataset.csv'
    return PATH, RANDOM_STATE


@app.cell
def _(PATH):
    # Loading datasets

    SleepDataset = pd.read_csv(
        PATH.format('Clean'),
        index_col = 0,
    )

    ProcessedSleepDataset = pd.read_csv(
        PATH.format('Processed'),
        index_col = 0,
    )
    return ProcessedSleepDataset, SleepDataset


@app.cell
def _(ProcessedSleepDataset, SleepDataset):
    # Splitting features

    TargetLabel = 'Quality of Sleep'
    Features = [_label for _label in SleepDataset.columns if _label != TargetLabel]
    ProcessedFeatures = [_label for _label in ProcessedSleepDataset.columns if _label != TargetLabel]
    return


@app.cell
def _():
    mo.md(r"# Data Mining")
    return


@app.cell
def _():
    mo.md(r"## 1. Cluster Analysis")
    return


@app.cell
def _():
    mo.md(r"")
    return


@app.function
def PlotSilhouetteResults(
        ValuesCriterion: list[int|float],
        SilhouetteScores: list[float],
        CriterionName: str,
        ClusteringTechnique: str,
    ):

    fig , axes = plt.subplots(
        subplot_kw = {'frame_on':False},
    )

    sns.lineplot(
        x = ValuesCriterion,
        y = SilhouetteScores,
        color = src.BaseColor,
        linestyle = '--',
        linewidth = 1.5,
        marker = 'o',
        markersize = 6,
        ax = axes,
    )
    axes.set_xlabel(CriterionName,size=12)
    axes.set_ylabel('Inertia value',size=12)
    axes.set_title(f'Scree Plot for Selection of\n{CriterionName} in {ClusteringTechnique}',size=14)
    axes.tick_params(axis='both',labelsize=10)

    return fig


@app.cell
def _(ProcessedSleepDataset, RANDOM_STATE):
    # Calculating and plotting Silhouette scores for K-Means

    _DimensionalReduction = PCA(n_components=4,whiten=True,random_state=RANDOM_STATE)
    _Clustering = KMeans(random_state=RANDOM_STATE)
    _TransformedDataset = _DimensionalReduction.fit_transform(ProcessedSleepDataset)

    _MaxNumClusters = 10
    _SilhouetteResults = []
    for _num_clusters in range(2,_MaxNumClusters+1):
        _Clustering.set_params(n_clusters=_num_clusters)
        _labels_clusters = _Clustering.fit_predict(ProcessedSleepDataset)

        _score = silhouette_score(_TransformedDataset,_labels_clusters)
        _SilhouetteResults.append(_score)

    PlotSilhouetteResults(
        range(2,_MaxNumClusters+1),
        _SilhouetteResults,
        'Number of Clusters',
        'K Means'
    )
    return


@app.cell
def _(ProcessedSleepDataset, RANDOM_STATE):
    # Calculating and plotting Silhouette scores for Single Agglomerative Clustering

    _DimensionalReduction = PCA(n_components=4,whiten=True,random_state=RANDOM_STATE)
    _Clustering = AgglomerativeClustering(linkage='single')
    _TransformedDataset = _DimensionalReduction.fit_transform(ProcessedSleepDataset)

    _MaxNumClusters = 10
    _SilhouetteResults = []
    for _num_clusters in range(2,_MaxNumClusters+1):
        _Clustering.set_params(n_clusters=_num_clusters)
        _labels_clusters = _Clustering.fit_predict(ProcessedSleepDataset)

        _score = silhouette_score(_TransformedDataset,_labels_clusters)
        _SilhouetteResults.append(_score)

    PlotSilhouetteResults(
        range(2,_MaxNumClusters+1),
        _SilhouetteResults,
        'Number of Clusters',
        'Single Agglomerative'
    )
    return


@app.cell
def _(ProcessedSleepDataset, RANDOM_STATE):
    # Calculating and plotting Silhouette scores for Complete Agglomerative Clustering

    _DimensionalReduction = PCA(n_components=4,whiten=True,random_state=RANDOM_STATE)
    _Clustering = AgglomerativeClustering(linkage='complete')
    _TransformedDataset = _DimensionalReduction.fit_transform(ProcessedSleepDataset)

    _MaxNumClusters = 10
    _SilhouetteResults = []
    for _num_clusters in range(2,_MaxNumClusters+1):
        _Clustering.set_params(n_clusters=_num_clusters)
        _labels_clusters = _Clustering.fit_predict(ProcessedSleepDataset)

        _score = silhouette_score(_TransformedDataset,_labels_clusters)
        _SilhouetteResults.append(_score)

    PlotSilhouetteResults(
        range(2,_MaxNumClusters+1),
        _SilhouetteResults,
        'Number of Clusters',
        'Complete Agglomerative'
    )
    return


@app.cell
def _(ProcessedSleepDataset, RANDOM_STATE):
    # Calculating and plotting Silhouette scores for DBSCAN

    _DimensionalReduction = PCA(n_components=4,whiten=True,random_state=RANDOM_STATE)
    _Clustering = DBSCAN()
    _TransformedDataset = _DimensionalReduction.fit_transform(ProcessedSleepDataset)

    _SilhouetteResults = []
    for _num_clusters in np.linspace(0.1,1,10):
        _Clustering.set_params(eps=_num_clusters)
        _labels_clusters = _Clustering.fit_predict(ProcessedSleepDataset)

        _score = silhouette_score(_TransformedDataset,_labels_clusters)
        _SilhouetteResults.append(_score)

    PlotSilhouetteResults(
        np.linspace(0.1,1,10),
        _SilhouetteResults,
        'Epsilon',
        'DBSCAN'
    )
    return


if __name__ == "__main__":
    app.run()
