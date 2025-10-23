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
    from sklearn.preprocessing import scale
    from sklearn.pipeline import Pipeline

    from sklearn.metrics import silhouette_score , mutual_info_score


    # Importing Functions and Utils
    import SourceStatisticalAnalysis as src


@app.cell
def _():
    mo.md(r"##")
    return


@app.cell
def _():
    # Setting constants

    PATH = src.PATH + '{}SleepDataset.csv'
    return (PATH,)


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
    return ProcessedFeatures, TargetLabel


@app.cell
def _():
    mo.md(r"# Data Mining")
    return


@app.cell
def _():
    mo.md(r"")
    return


@app.cell
def _():
    mo.md(r"## 1. Cluster Analysis")
    return


@app.cell
def _():
    mo.md(
        r"""
        Using the processed dataset with encoded categorical values, K Means and Agglomerative clustering (both single and complete) are employed in order to evaluate which algorithm generates the best results using Euclidean distance as the metric (due to the presence of mixed data).
    
        No scaling transformation was performed, even though the features belong to different scales, because it was noted that rescaling "distorts" the distances between data points enough to lose part of the inherent structure they possess, that is, the groups they form and their natural separability.
    
        Due to how each clustering algorithm works, Silhouette score was chosen to measure the quality of the algorithms and compare them based on how well they separate and generate clusters. Using the elbow method, it is found that similar results are reached where the optimal number of clusters is $6$ along with comparable scores.
        """
    )
    return


@app.cell
def _(ProcessedFeatures):
    # Creating selectors of features for plotting the values

    FeatureOptionsCluster_1 = mo.ui.dropdown(
        ProcessedFeatures,
        value = ProcessedFeatures[0],
        label = 'Select a Feature',
        allow_select_none = False,
    )

    FeatureOptionsCluster_2 = mo.ui.dropdown(
        ProcessedFeatures,
        value = ProcessedFeatures[0],
        label = 'Select a Feature',
        allow_select_none = False,
    )
    return FeatureOptionsCluster_1, FeatureOptionsCluster_2


@app.cell
def _(ProcessedFeatures, ProcessedSleepDataset, TargetLabel):
    # Splitting dataset into features and target values

    ProcessedSleepDataset_Features = ProcessedSleepDataset[ProcessedFeatures]
    ProcessedSleepDataset_ScaledFeatures = scale(ProcessedSleepDataset_Features)
    ProcessedSleepDataset_Target = ProcessedSleepDataset[TargetLabel]

    DatasetClustering = ProcessedSleepDataset_Features
    return (
        DatasetClustering,
        ProcessedSleepDataset_Features,
        ProcessedSleepDataset_Target,
    )


@app.cell
def _(
    FeatureOptionsCluster_1,
    FeatureOptionsCluster_2,
    ProcessedSleepDataset_Features,
):
    _fig , _axes = src.CreatePlot()

    _Feature_1 = FeatureOptionsCluster_1.value
    _Feature_2 = FeatureOptionsCluster_2.value
    sns.scatterplot(
        x = ProcessedSleepDataset_Features[_Feature_1],
        y = ProcessedSleepDataset_Features[_Feature_2],
        c = src.BaseColor,
        ax = _axes
    )
    src.SetLabelsToPlot(
        _axes,
        f'{_Feature_1} vs {_Feature_2}',
        _Feature_1,
        _Feature_2,
    )

    mo.vstack(
        [
            mo.hstack([FeatureOptionsCluster_1,FeatureOptionsCluster_2]),
            _fig 
        ]
    )
    return


@app.cell
def _(DatasetClustering):
    # Calculating and plotting Silhouette scores for K-Means

    ClusteringKMeans = KMeans(random_state=src.RANDOM_STATE)

    _MaxNumClusters = 10
    _SilhouetteResults = []
    for _num_clusters in range(2,_MaxNumClusters+1):
        ClusteringKMeans.set_params(n_clusters=_num_clusters)
        _labels_clusters = ClusteringKMeans.fit_predict(DatasetClustering)

        _score = silhouette_score(DatasetClustering,_labels_clusters)
        _SilhouetteResults.append(_score)

    _fig = src.PlotSilhouetteResults(
        range(2,_MaxNumClusters+1),
        _SilhouetteResults,
        'Number of Clusters',
        'K Means'
    )

    _fig
    return (ClusteringKMeans,)


@app.cell
def _(DatasetClustering):
    # Calculating and plotting Silhouette scores for Single Agglomerative Clustering

    ClusteringAgglomerativeSingle = AgglomerativeClustering(linkage='single')

    _MaxNumClusters = 10
    _SilhouetteResults = []
    for _num_clusters in range(2,_MaxNumClusters+1):
        ClusteringAgglomerativeSingle.set_params(n_clusters=_num_clusters)
        _labels_clusters = ClusteringAgglomerativeSingle.fit_predict(DatasetClustering)

        _score = silhouette_score(DatasetClustering,_labels_clusters)
        _SilhouetteResults.append(_score)

    _fig = src.PlotSilhouetteResults(
        range(2,_MaxNumClusters+1),
        _SilhouetteResults,
        'Number of Clusters',
        'Single Agglomerative'
    )

    _fig
    return (ClusteringAgglomerativeSingle,)


@app.cell
def _(DatasetClustering):
    # Calculating and plotting Silhouette scores for Complete Agglomerative Clustering

    ClusteringAgglomerativeComplete = AgglomerativeClustering(linkage='complete')

    _MaxNumClusters = 10
    _SilhouetteResults = []
    for _num_clusters in range(2,_MaxNumClusters+1):
        ClusteringAgglomerativeComplete.set_params(n_clusters=_num_clusters)
        _labels_clusters = ClusteringAgglomerativeComplete.fit_predict(DatasetClustering)

        _score = silhouette_score(DatasetClustering,_labels_clusters)
        _SilhouetteResults.append(_score)

    _fig = src.PlotSilhouetteResults(
        range(2,_MaxNumClusters+1),
        _SilhouetteResults,
        'Number of Clusters',
        'Complete Agglomerative'
    )

    _fig
    return (ClusteringAgglomerativeComplete,)


@app.cell
def _():
    mo.md(r"Using mutual information score to compare the labels assigned by the clustering algorithms and the ground truth labels of the target (`Quality of Sleep`), the three algorithms have similar and significant scores. This indicates that the processed data (and therefore the original data) form clusters, that is, there exists some pattern surrounding the sleep quality of patients that allows the data to have a structure enabling the generation of potential profiles and descriptions of patients based on their factors and habits.")
    return


@app.cell
def _(
    ClusteringAgglomerativeComplete,
    ClusteringAgglomerativeSingle,
    ClusteringKMeans,
    DatasetClustering,
    ProcessedSleepDataset_Target,
):
    # Evaluating each clustering algorithm with the best number of clusters

    _ClusteringAlgorithms = [
        (ClusteringKMeans,'K Means'),
        (ClusteringAgglomerativeSingle,'Single Agglomerative'),
        (ClusteringAgglomerativeComplete,'Complete Agglomerative'),
    ]

    _TableEvaluationResults = []
    for _clustering , _name in _ClusteringAlgorithms:
        _clustering.set_params(n_clusters=6)
        _labels_clusters = _clustering.fit_predict(DatasetClustering)

        _score = mutual_info_score(ProcessedSleepDataset_Target,_labels_clusters)
        _TableEvaluationResults.append([_name,_score])

    mo.vstack(
        [
            mo.md('**Mutual Information Scores With 6 Clusters**'),
            pd.DataFrame(_TableEvaluationResults,columns=['Clustering Algorithm','MI Score']),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
