import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium", app_title="Data Mining")

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
    from gower import gower_matrix

    from sklearn.preprocessing import StandardScaler , MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import AgglomerativeClustering , KMeans , DBSCAN
    from sklearn.preprocessing import scale
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import silhouette_score , mutual_info_score

    from mlxtend.frequent_patterns import apriori, association_rules


    # Importing Functions and Utils
    import SourceStatisticalAnalysis as src


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
    return Features, ProcessedFeatures, TargetLabel


@app.cell
def _():
    mo.Html(src.HeaderNav)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Data Mining
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Derived from the statistical analysis performed, different patterns and associations were detected that can be clarified or discovered using Data Mining techniques. This notebook explores and discusses the results of these techniques (Cluster Analysis and Association Rules), which allow for the verification of the observations made during the EDA.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Cluster Analysis
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### 1.1. Visualization of the Dataset
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Using PCA in the processed dataset (encoded categorical features and MinMax scaler for numerical features, this transformations allow to have a same scale), no clusters were found visually. This implies that it is necessary to apply feature engineering to create another kind of relationships between features, and using an appropriate metric could show better clusters in the dataset. However, the plots do not show the emergence of well-defined clusters.
    """)
    return


@app.cell
def _(
    Features,
    ProcessedFeatures,
    ProcessedSleepDataset,
    SleepDataset,
    TargetLabel,
):
    # Splitting dataset into features and target values

    ProcessedSleepDataset_Features = ProcessedSleepDataset[ProcessedFeatures]
    ProcessedSleepDataset_Target = ProcessedSleepDataset[TargetLabel]

    # Creating distance matrix with Gower Distance

    DatasetClustering = gower_matrix(SleepDataset[Features])
    return (
        DatasetClustering,
        ProcessedSleepDataset_Features,
        ProcessedSleepDataset_Target,
    )


@app.cell
def _(
    ProcessedSleepDataset_Features,
    ProcessedSleepDataset_Target,
    TargetLabel,
):
    # Applying PCA to the preprocessed dataset

    _PipelinePCA = Pipeline(
        [
            ('Standardization',MinMaxScaler()),
            ('PrincipalComponents',PCA(random_state=src.RANDOM_STATE))
        ]
    )
    SleepDatasetReducedPCA = _PipelinePCA.fit_transform(ProcessedSleepDataset_Features)

    _fig , _axes = plt.subplot_mosaic(
        '1122\n.33.',
        subplot_kw = {'frame_on':False},
        layout = 'constrained',
        figsize = (6,5),
    )

    _legend_handles = None
    for (_pc_x , _pc_y) , _index_ax in zip(combinations(range(3),2),range(1,4)):
        _ax = _axes[str(_index_ax)]
        sns.scatterplot(
            x = SleepDatasetReducedPCA[:,_pc_x],
            y = SleepDatasetReducedPCA[:,_pc_y],
            hue = ProcessedSleepDataset_Target,

            palette = src.BasePalette(n_colors=6),
            ax = _ax,
        )
        src.SetLabelsToPlot(
            _ax,
            None,
            f'PC {_pc_x+1}',
            f'PC {_pc_y+1}',
            None,9,8
        )

        _legend_handles = _ax.get_legend_handles_labels()
        _ax.legend_ = None

    src.SetFigureTitle(
        _fig,
        'Dimensionality Reduction with PCA',
        13,
    )

    _fig.legend(
        *_legend_handles,
        title = TargetLabel,
        loc = 'lower right'
    )

    # _fig.savefig(f'../Resources/Clustering_ClusterPlots.jpg')
    _fig
    return


@app.cell
def _():
    mo.md(r"""
    ### 1.2. Applying Clustering with Different Techniques
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Agglomerative clustering is employed in order to evaluate which linkage generates the best results using Gower distance as the metric (due to the presence of mixed data K-Means can not be used).

    Due to how each linkage algorithm works, Silhouette score was chosen to measure the quality of the linkages and to compare them based on how well they separate and generate clusters. Using the scree plots of the Silhouette score varying the number of clusters, the complete linkage tends to have better scores due to this linkage forms sphere-shape clusters. Therefore, clusters with patients more similar to each other.

    Using the elbow method, agglomerative clustering with complete linkage and 8 clusters achieves a Silhouette score of 0.5737 which each cluster is well-defined and represents a distinct profile with possible similarities.
    """)
    return


@app.cell
def _(DatasetClustering):
    # Calculating and plotting Silhouette scores for Single Agglomerative Clustering

    ClusteringAgglomerativeSingle = Pipeline(
        [
            ('Clustering',AgglomerativeClustering(linkage='single',metric='precomputed')),
        ]
    )

    _MaxNumClusters = 10
    _SilhouetteResults = []
    for _num_clusters in range(2,_MaxNumClusters+1):
        ClusteringAgglomerativeSingle.set_params(Clustering__n_clusters=_num_clusters)
        _labels_clusters = ClusteringAgglomerativeSingle.fit_predict(DatasetClustering)

        _score = silhouette_score(DatasetClustering,_labels_clusters,metric='precomputed')
        _SilhouetteResults.append(_score)

    _fig = src.PlotSilhouetteResults(
        range(2,_MaxNumClusters+1),
        _SilhouetteResults,
        'Number of Clusters',
        'Single Agglomerative'
    )

    _fig
    return


@app.cell
def _(DatasetClustering):
    # Calculating and plotting Silhouette scores for Complete Agglomerative Clustering

    ClusteringAgglomerativeComplete = Pipeline(
        [
            ('Clustering',AgglomerativeClustering(linkage='complete',metric='precomputed')),
        ]
    )

    _MaxNumClusters = 10
    _SilhouetteResults = []
    for _num_clusters in range(2,_MaxNumClusters+1):
        ClusteringAgglomerativeComplete.set_params(Clustering__n_clusters=_num_clusters)
        _labels_clusters = ClusteringAgglomerativeComplete.fit_predict(DatasetClustering)

        _score = silhouette_score(DatasetClustering,_labels_clusters,metric='precomputed')
        _SilhouetteResults.append(_score)

    _fig = src.PlotSilhouetteResults(
        range(2,_MaxNumClusters+1),
        _SilhouetteResults,
        'Number of Clusters',
        'Complete Agglomerative'
    )

    # _fig.savefig(f'../Resources/Clustering_ScreePlotComplete.jpg')
    _fig
    return (ClusteringAgglomerativeComplete,)


@app.cell
def _(DatasetClustering):
    # Calculating and plotting Silhouette scores for Complete Agglomerative Clustering

    ClusteringAgglomerativeAverage = Pipeline(
        [
            ('Clustering',AgglomerativeClustering(linkage='average',metric='precomputed')),
        ]
    )

    _MaxNumClusters = 10
    _SilhouetteResults = []
    for _num_clusters in range(2,_MaxNumClusters+1):
        ClusteringAgglomerativeAverage.set_params(Clustering__n_clusters=_num_clusters)
        _labels_clusters = ClusteringAgglomerativeAverage.fit_predict(DatasetClustering)

        _score = silhouette_score(DatasetClustering,_labels_clusters,metric='precomputed')
        _SilhouetteResults.append(_score)

    _fig = src.PlotSilhouetteResults(
        range(2,_MaxNumClusters+1),
        _SilhouetteResults,
        'Number of Clusters',
        'Average Agglomerative'
    )

    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 1.3. Profiles of Patients
    """)
    return


@app.cell
def _(SleepDataset):
    SleepDataset
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Because of complete linkage has the best Silhouette score, it is chosen for the profiles and their respective description are generated based on its results of clustering (cluster centers, centroids). The next profiles are discovered:

    * **Profile 1**: Predominantly male subjects aged 42 years with normal sleep quality and an average sleep duration of 6.5 hours. These individuals exhibit low to moderate physical activity and overweight status, alongside moderately elevated stress levels, frequently manifesting as insomnia. This profile likely represents middle-aged professionals engaged in demanding office-based roles, characterized by significant time commitment.

    * **Profile 2**: Male subjects aged 40 years with good sleep quality and an average sleep duration of 7.5 hours. These individuals demonstrate high levels of physical activity and maintain a normal body weight, accompanied by normal stress levels and a lack of sleep disorders. This profile likely comprises middle-aged professionals with elevated socioeconomic status and employment roles requiring reduced time commitment, facilitating a higher quality of life.

    * **Profile 3**: Predominantly male subjects aged 38 years with normal sleep quality and an average sleep duration of 7 hours. These individuals exhibit normal physical activity levels and are classified as obese, alongside normal-to-elevated stress levels, frequently experiencing sleep disorders, including obstructive sleep apnea or insomnia secondary to obesity. This profile likely represents young adult males attempting to integrate physical activity into their daily routines despite obesity-related sleep challenges.

    * **Profile 4**: Predominantly female subjects aged 36 years with good sleep quality and an average sleep duration of 7.2 hours. These individuals demonstrate moderate physical activity and maintain a normal body weight, accompanied by low-to-normal stress levels and a lack of sleep disorders. This profile likely comprises young professional women holding positions affording a degree of work-life balance.

    * **Profile 5**: Male subjects aged 30 years with normal sleep quality and an average sleep duration of 6.8 hours. These individuals exhibit normal physical activity levels and maintain a normal body weight, alongside elevated stress levels and a lack of sleep disorders. This profile likely represents young male doctors initiating their professional careers, characterized by demanding work schedules and associated stress.

    * **Profile 6**: Female subjects aged 52 years with excellent sleep quality and an average sleep duration of 8.4 hours. These individuals demonstrate low levels of physical activity and maintain a normal body weight, accompanied by low stress levels and a lack of sleep disorders. This profile likely comprises established professional women or those with high socioeconomic status, facilitating a quiet lifestyle.

    * **Profile 7**: Female subjects aged 43 years with normal-good sleep quality and an average sleep duration of 6.5 hours. These individuals demonstrate moderate physical activity and are classified as overweight, alongside normal-to-elevated stress levels, frequently manifesting as insomnia. This profile likely represents adult female educators whose profession demands significant time and energy, yet who incorporate physical activity into their routines.

    * **Profile 8**: Female subjects aged 53 years with good sleep quality and an average sleep duration of 7 hours. These individuals demonstrate high levels of physical activity and are classified as overweight, alongside normal stress levels, frequently experiencing obstructive sleep apnea. This profile likely comprises experienced nurses who have established their professional routines and maintain a normal lifestyle without significant worries.
    """)
    return


@app.cell
def _(
    ClusteringAgglomerativeComplete,
    DatasetClustering,
    ProcessedSleepDataset,
):
    # Calculating values for feature and target in each profile

    _num_clusters = 8
    ClusteringAgglomerativeComplete.set_params(Clustering__n_clusters=_num_clusters)
    ClusteringLabels = ClusteringAgglomerativeComplete.fit_predict(DatasetClustering)

    ClusterProfiles = ProcessedSleepDataset.groupby(ClusteringLabels).mean()

    ClusterProfiles.index = [f'Profile {_profile}' for _profile in range(1,_num_clusters+1)]
    ClusterProfiles = ClusterProfiles.T
    return (ClusterProfiles,)


@app.cell
def _(ClusterProfiles):
    mo.vstack(
        [
            mo.md('**Profiles for Each Cluster**'),
            ClusterProfiles.map(src.OutputFormatting),
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Frequent Patterns And Association Rules
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### 2.1. Discretization of Numerical Values
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In order to apply pattern extraction techniques, the data must be binary; that is, each feature must represent the absence or presence of a certain property. Therefore, the numerical features first had to be categorized (trough creating value ranges) before applying One Hot Encoding to each of them to obtain their respective binary features (values).

    For features such as `Age`, `Sleep Duration`, `Heart Rate`, `Daily Steps`, and `Blood Pressure`, the official value ranges were researched according to recognizable health organizations. For the remaining features, ranges were created based on the values they take and their descriptions according to the metadata of the dataset (these ranges are more arbitrary).
    """)
    return


@app.cell
def _(ProcessedSleepDataset, SleepDataset):
    # Discretization of numerical values 
    # (transformation into categorical features)

    EncodedSleepDataset = ProcessedSleepDataset.copy(True)

    EncodedSleepDataset.rename(columns={'Gender':'Male'},inplace=True)

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


@app.cell
def _():
    mo.md(r"""
    ### 2.2. Association Rules
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Based on the results obtained from the frequent patterns, using a minimum support of 15% (approximately 56 patients), association rules are derived whose confidence is greater than 90% and a lift greater than 5 to ensure they are strong rules and non-random behaviors. The rules discovered and considered relevant for the study are the following:
    """)
    return


@app.cell
def _():
    _RelevantRules = pd.DataFrame(
        [
            ['Sleep Duration = Optimal , Age = Middle-Aged , Stress Level = Low','Heart Rate = Normal , Quality of Sleep = Excellent'],
            ['BMI Category = Overweight , Sleep Disorder = Insomnia','Sleep Duration = Short , Daily Steps = Low , Heart Rate = Normal , Physical Activity Level = Sedentary'],
            ['Blood Pressure  = Hypertension Stage 2, Occupation = Nurse','Sleep Disorder = Sleep Apnea , BMI Category = Overweight'],
        ],
        columns = ['Antecedent','Consequent']
    )

    mo.vstack(
        [
            mo.md('**Relevant Association Rules**'),
            _RelevantRules,
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The first rule reflects the conditions for achieving the best rest/sleep and having a normal heart rate, which are having low stress levels, sleeping between 7 and 9 hours, and being between 50 and 60 years old. Overall, this rule explains how sleeping well and not living stressed impacts how well one sleeps.

    The second rule shows the association that exists between having a precarious health status (overweight and insomnia) and the habits of a patient. The most relevant finding is that it verifies the pattern that being overweight implies being sedentary and having low physical activity, and that this adds to the occurrence of insomnia, resulting in fewer hours of sleep.

    The third rule appears as a particular case within the professional life of nurses, where those suffering from hypertension tend to be overweight and have apnea. This fact can be verified by considering that the onset of hypertension is associated with being overweight and poor habits, and that apnea is frequent among overweight individuals.
    """)
    return


@app.cell
def _(EncodedSleepDataset):
    # Searching frequent patterns on the dataset

    BooleanSleepDataset = EncodedSleepDataset.astype(bool)

    FrequentPatterns = apriori(
        BooleanSleepDataset,
        min_support = 0.15,
        use_colnames = True,
    )
    FrequentPatterns['itemsets'] = FrequentPatterns['itemsets'].apply(list)

    mo.vstack(
        [
            mo.md('**Frequent Patterns**'),
            FrequentPatterns,
        ]
    )
    return (FrequentPatterns,)


@app.cell
def _(FrequentPatterns):
    # Generation of association rules

    AssociationRules = association_rules(
        FrequentPatterns,
        metric = 'lift',
        min_threshold = 1,
    )
    AssociationRules['antecedents'] = AssociationRules['antecedents'].apply(list)
    AssociationRules['consequents'] = AssociationRules['consequents'].apply(list)

    _RelevantMetrics = ['support','confidence','lift',]
    AssociationRules.sort_values(_RelevantMetrics,ascending=False,inplace=True)
    AssociationRules.query("confidence > 0.9 & lift > 5",inplace=True)

    mo.vstack(
        [
            mo.md('**Association Rules**'),
            AssociationRules[['antecedents','consequents',*_RelevantMetrics]].map(partial(src.OutputFormatting,Precision=6)),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    #
    """)
    return


@app.cell
def _():
    mo.Html(src.Footer)
    return


if __name__ == "__main__":
    app.run()
