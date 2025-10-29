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

    # _fig.savefig(f'./Resources/ClusterAnalysis_ScreeKMeans.jpg')
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


@app.cell
def _():
    mo.md(r"### 1.1. Profiles of Patients")
    return


@app.cell
def _():
    mo.md(
        r"""
        Because of K-Means has the best Silhouette score and a high MI score, it is chosen for the profiles and their respective description are generated based on its results of clustering (cluster centers). The next profiles are discovered:
    
        * **Profile 1**: Women aged 46 with low stress levels and moderate physical activity, which is reflected in a life with normal blood pressure and heart rate, allowing them to sleep for sufficient time with good rest, and not suffer from insomnia, tend to have sleep apnea derived from a tendency to be overweight. Their main professions are accountants and nursing, fields that allow for a balanced lifestyle with low stress levels.
    
        * **Profile 2**: People between 43 and 44 years old with moderately stressful lives, getting 6.5 hours of daily sleep derived or caused by mostly suffering from insomnia, who have low physical activity resulting in overweight along with slightly above-normal blood pressure and heart rate. Their main professions are managers and teachers, fields with constant work pressure that consume most of their time.
    
        * **Profile 3**: Women between 48 and 49 years old with deplorable sleep quality and rest derived from suffering from sleep apnea, which causes a highly stressful life with arrhythmias (high blood pressure and heart rate), they have high levels of physical activity which benefits their overall condition. They are mostly nurses, a field where sleep hours are low and work shifts are stressful.
    
        * **Profile 4**: Men aged 36 with moderately stressful lives that allow them to have ideal rest and recovery, engage in some physical activity, which is explained by considering they don't suffer from sleep disorders or overweight/obesity. They are mostly doctors and lawyers by profession, fields that do involve stress but once they achieve a stable position allow for a more controlled life.
    
        * **Profile 5**: Mostly men aged 35 with deplorable sleep quality derived from suffering from sleep disorders that result in less willingness to engage in physical activity and worse quality of life (greater tendency to be overweight), which also leads to higher than normal blood pressure and heart rate. They are mostly software engineers and sales representatives, two fields that require high time demands and constant workload.
    
        * **Profile 6**: People between 42 and 43 years old with moderate sleep quality living lives with little physical activity but without sleep disorders or overweight, both their blood pressure and heart rate are slightly above normal but not alarming. They are mostly engineers and scientists, fields that limit time dedicated to physical and recreational activities.
        """
    )
    return


@app.cell
def _(
    ClusteringKMeans,
    DatasetClustering,
    ProcessedSleepDataset_Target,
    TargetLabel,
):
    # Calculating values for feature and target in each profile

    ClusteringKMeans.set_params(n_clusters=6)
    ClusteringLabels = ClusteringKMeans.fit_predict(DatasetClustering)

    ClusterProfiles = pd.DataFrame(
        ClusteringKMeans.cluster_centers_,
        columns = DatasetClustering.columns,
    )
    ClusterProfiles[TargetLabel] = ProcessedSleepDataset_Target.groupby(ClusteringLabels).mean()

    ClusterProfiles.index = [f'Profile {_profile}' for _profile in range(1,7)]
    ClusterProfiles = ClusterProfiles.T
    return (ClusterProfiles,)


@app.cell
def _(ClusterProfiles):
    mo.vstack(
        [
            mo.md('**Profiles for Each Cluster**'),
            ClusterProfiles,
        ]
    )
    return


@app.cell
def _(ClusterProfiles):
    MaxMinClusterProfiles = pd.concat(
        [
            ClusterProfiles.apply(np.argmin,axis=1),
            ClusterProfiles.apply(np.argmax,axis=1),
        ],axis=1,
    ) + 1
    MaxMinClusterProfiles.columns = ['Min Value Profile','Max Value Profile']

    mo.vstack(
        [
            mo.md('**Minimum and Maximum Values on each Feature**'),
            MaxMinClusterProfiles,
        ]
    )
    return


@app.cell
def _():
    mo.md(r"## 2. Patterns And Association Rules")
    return


@app.cell
def _():
    mo.md(
        r"""
        In order to apply pattern extraction techniques, the data must be binary; that is, each feature must represent the absence or presence of a certain property. Therefore, the numerical features first had to be categorized (trough creating value ranges) before applying One Hot Encoding to each of them to obtain their respective binary features (values).
    
        For features such as `Age`, `Sleep Duration`, `Heart Rate`, `Daily Steps`, and `Blood Pressure`, the official value ranges were researched according to recognizable health organizations. For the remaining features, ranges were created based on the values they take and their descriptions according to the metadata of the dataset (these ranges are more arbitrary).
        """
    )
    return


@app.cell
def _(ProcessedSleepDataset, SleepDataset):
    # Discretization of numerical values 
    # (transformation into categorical features)

    EncodedSleepDataset = ProcessedSleepDataset.copy(True)

    EncodedSleepDataset['BMI Category'] = SleepDataset['BMI Category']
    src.OneHotEncoderFeature(
        EncodedSleepDataset,'BMI Category',
        None,None,
    )

    src.OneHotEncoderFeature(
        EncodedSleepDataset,'Age',[25,30,50,60],
        ['Young Adult','Adult','Middle-Aged'],
    )

    src.OneHotEncoderFeature(
        EncodedSleepDataset,'Sleep Duration',
        [5,6,7,9],['Lack','Short','Optimal'],
    )

    src.OneHotEncoderFeature(
        EncodedSleepDataset,'Quality of Sleep',
        [3,6,8,10],['Low','Regular','Excellent'],
    )

    src.OneHotEncoderFeature(
        EncodedSleepDataset,'Physical Activity Level',
        [20,45,60,75,100],['Sedentary','Low','Moderate','High'],
    )
 
    src.OneHotEncoderFeature(
        EncodedSleepDataset,'Stress Level',
        [2,5,7,9],['Low','Moderate','High'],
    )
                
    src.OneHotEncoderFeature(
        EncodedSleepDataset,'Heart Rate',
        [60,80,100],['Normal','High'],
    )
                
    src.OneHotEncoderFeature(
        EncodedSleepDataset,'Daily Steps',
        [2000,5000,7000,8000,11000],
        ['Sedentary','Low','Moderate','High'],
    )

    EncodedSleepDataset['Blood Pressure'] = EncodedSleepDataset[['Blood Pressure Systolic','Blood Pressure Diastolic']].apply(lambda blood_pressure: src.CategorizeBloodPressure(*blood_pressure),axis=1)
    EncodedSleepDataset.drop(columns=['Blood Pressure Systolic','Blood Pressure Diastolic'],inplace=True)
    src.OneHotEncoderFeature(
        EncodedSleepDataset,'Blood Pressure',
        None,None,
    )
    return (EncodedSleepDataset,)


@app.cell
def _(EncodedSleepDataset):
    mo.vstack(
        [
            mo.md('**Examples of Encoded Records**'),
            EncodedSleepDataset,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
