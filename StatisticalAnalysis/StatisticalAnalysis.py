import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium", app_title="Statistical Analysis")

with app.setup:
    # Import auxiliar libraries
    import marimo as mo
    from itertools import combinations , product
    from functools import partial


    # Importing libraries
    import pandas as pd
    import numpy as np

    import seaborn as sns
    import matplotlib.pyplot as plt

    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.multivariate.factor import Factor

    from scipy import stats

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OrdinalEncoder , OneHotEncoder 
    from sklearn.pipeline import Pipeline

    from mlxtend.feature_selection  import SequentialFeatureSelector


    # Importing Functions and Utils
    import SourceStatisticalAnalysis as src


@app.cell
def _():
    mo.Html(src.HeaderNav)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Exploratory Data Analysis

    Sleep quality may be influenced by habits and lifestyle of a subject; therefore, understanding how these values and measures influence a subject could determine how well they sleep. Thus, this study examines how various factors that determine lifestyle of a subject can impact their sleep quality.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Load Dataset and First Exploration
    """)
    return


@app.cell
def _():
    mo.md(r"""
    The dataset is taken from [Health and Sleep relation](https://www.kaggle.com/datasets/orvile/health-and-sleep-relation-2024). This dataset explores the relationship between sleep patterns and overall health. It includes detailed information on individual sleep habits, health metrics, and lifestyle factors, enabling analysis of how sleep quality and duration impact physical and mental well-being.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The dataset contains `374` instances, with `12` attributes which describe the patient's sleep health. These columns (attributes) are:

    * `Gender` (*Nominal*): The patient's biological sex (Male or Female)

    * `Age` (*Discrete*): The patient's age

    * `Occupation` (*Nominal*): The patient's current job title (e.g., Doctor, Engineer, Teacher)

    * `Sleep Duration` (*Continuous*): The average time spent sleeping per night, measured in hours

    * `Quality of Sleep` (*Discrete*): A subjective rating of sleep quality

    * `Physical Activity Level` (*Discrete*): A subjective rating of physical activity

    * `Stress Level` (*Discrete*): The patient's perceived stress level

    * `BMI Category` (*Ordinal*): The patient's Body Mass Index classification (Normal, Overweight, etc.)

    * `Blood Pressure` (*Ordinal*): The measure of blood pressure, provided as a Systolic/Diastolic format

    * `Heart Rate` (*Discrete*): The patient's average heart rate, measured in Beats Per Minute (BPM)

    * `Daily Steps` (*Discrete*): The total number of steps taken by the patient on a daily basis

    * `Sleep Disorder` (*Nominal*): Indicates the presence or absence of a diagnosed sleep disorder, such as Sleep Apnea or Insomnia

    There are missing values on `Sleep Disorder` because of there are patients without sleep disorders.  The notation of `Blood Pressure` is Systolic/Diastolic form which will be transformed. There is a duplicate category in `BMI Category` (Normal and Normal Weight) which one of them will be removed. `Quality of Sleep` is the feature to predict (target) and study in this analysis.
    """)
    return


@app.cell
def _():
    # Loading dataset

    _RawDataset = 'Sleep_health_and_lifestyle_dataset.csv'
    SleepDataset_Raw = pd.read_csv(
        src.PATH + _RawDataset,
        index_col = 0,
    )
    return (SleepDataset_Raw,)


@app.cell
def _(SleepDataset_Raw):
    mo.vstack(
        [
            mo.md("**Example of Instances**"),
            mo.plain(SleepDataset_Raw),
        ], 
        align = 'center',
    )
    return


@app.cell
def _(SleepDataset_Raw):
    mo.vstack(
        [
            mo.md("**Missing Values**"),
            SleepDataset_Raw.isna().sum(axis = 0),
        ], 
        align = 'center',
    )
    return


@app.cell
def _(SleepDataset_Raw):
    _MissingValuesInSleepDisorder = SleepDataset_Raw.groupby(
        'Sleep Disorder',
        dropna = False,
    )['Gender'].count()

    mo.vstack(
        [
            mo.md("**Missing Values In `Sleep Disorder`**"),
            _MissingValuesInSleepDisorder,
        ], 
        align = 'center',
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Data Transformation
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After a minor data exploration, some features have missing or weird values. Therefore, based on the context of each feature, the following transformations are applied:

    The missing values of `Sleep Disorder` are imputed with `No`, the values of `Blood Pressure` are splitted into systolic and diastolic values, and the values with `Normal Weight` in `BMI Category` are transformed to `Normal`.

    For a better understanding and interpretation of the target, the values of `Quality of Sleep` are rearranging from 1 to 6.
    """)
    return


@app.cell
def _(SleepDataset_Raw):
    SleepDataset = SleepDataset_Raw.copy()

    # Filling missing values

    SleepDataset['Sleep Disorder'] = SleepDataset['Sleep Disorder'].fillna('No')

    # Splitting blood pressure into systolic and diastolic

    SleepDataset[['Blood Pressure Systolic','Blood Pressure Diastolic']] = [*SleepDataset['Blood Pressure'].apply(src.SplitBloodPressure)]
    SleepDataset.drop(columns=['Blood Pressure'],inplace=True)

    # Removing duplicate category in BMI Category

    _IndexBMINormalWeight = SleepDataset.query("`BMI Category` == 'Normal Weight'").index
    SleepDataset.loc[_IndexBMINormalWeight,'BMI Category'] = 'Normal'

    # Rearranging values of Quality of Sleep

    SleepDataset['Quality of Sleep'] = SleepDataset['Quality of Sleep'] - 3

    # Saving a clean dataset

    try:
        SleepDataset.to_csv(
            src.PATH + 'CleanSleepDataset.csv'
        )
    except:
        pass
    return (SleepDataset,)


@app.cell
def _(SleepDataset):
    mo.vstack(
        [
            mo.md("**Example of Instances After Transformations**"),
            mo.plain(SleepDataset),
        ], 
        align = 'center',
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Univariate Analysis
    """)
    return


@app.cell
def _():
    mo.md(r"""
    In this section is performed a univariate analysis on both numerical and categorical features to gather initial insights based on the statistical properties and dataset's context. This analysis is focused on showing how the features impact on the patient's `Quality of Sleep`.
    """)
    return


@app.cell
def _(SleepDataset):
    # Splitting features into numerical and categorical features

    NumericalFeatures , CategoricalFeatures = src.SplitFeatures(SleepDataset)
    return CategoricalFeatures, NumericalFeatures


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 3.1. Numerical Features
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    None of the features are normal, so some of the techniques that will be used will lead to insignificant results. Therefore, the values could be transformed with power transformations like Box-Cox or it could be assumed that the results will be less significant. After using Box-Cox transformation there was no improve (the transformed distributions were still non-normal under Shapiro-Wilk test), therefore the analysis of the results using the techniques that will be used will be more detailed and thorough.

    Fifty percent of patients have a `Quality of Sleep` between 3 and 5, and a `Sleep Duration` of between 6.4 to 7.8 hours. This can be explained by considering that stress and physical activity influence sleep onset and the recovery of the body during sleep. To this, it can add the biological degradation of the body as a subject becomes older, which impacts the number of hours needed to feel rested after sleeping.

    The fifty percent of patients are middle-aged (between 35 and 50 years old), so this dataset has a high representative of a same generational age. This implies that many patients will have a similar lifestyle and habits (similar population), so if a model is trained with this dataset will be underfitted to predict the younger subjects' `Quality of Sleep`.
    """)
    return


@app.cell
def _(NumericalFeatures, SleepDataset):
    mo.vstack(
        [
            mo.md("**Statistics of Numerical Features**"),
            SleepDataset[NumericalFeatures].describe().iloc[1:].map(src.OutputFormatting),
        ], 
    )
    return


@app.cell
def _():
    # Creating selector of kind of plot

    KindPlotNumericalFeatures = mo.ui.dropdown(
        {'Violin':sns.violinplot,'Box':sns.boxplot,'Histogram':sns.histplot},
        value = 'Box',
        label = 'Choose a Kind of Plot: ',
    )
    return (KindPlotNumericalFeatures,)


@app.cell
def _(KindPlotNumericalFeatures, NumericalFeatures, SleepDataset):
    _fig , _axes = src.CreatePlot(
        3,3,
        (6,6)
    )

    for _ax , _feature in zip(_axes.ravel(),NumericalFeatures):
        KindPlotNumericalFeatures.value(
            SleepDataset,
            x = _feature,
            ax = _ax,
            color = src.BaseColor,
        )
        src.SetLabelsToPlot(
            _ax,
            None,
            _feature,
            _ax.get_ylabel(),
        )

    src.SetFigureTitle(_fig,'Distribution of Numerical Features')

    # _fig.savefig(f'../Resources/UnivariatePlot_Numerical.jpg')
    mo.vstack([KindPlotNumericalFeatures,_fig])
    return


@app.cell
def _(NumericalFeatures, SleepDataset):
    _DataShapiroResults = []
    for _numerical_feature in NumericalFeatures:
        _shapiro_result = stats.shapiro(SleepDataset[_numerical_feature])
        _DataShapiroResults.append((_numerical_feature,_shapiro_result.pvalue))

    mo.vstack(
        [
            mo.md("**P-Values of Shapiro-Wilk Tests for the Features**"),
            pd.DataFrame(_DataShapiroResults,columns=['Numerical Feature','P-Value']).map(src.OutputFormatting),
        ]
    )
    return


@app.cell
def _(NumericalFeatures, SleepDataset):
    _DataShapiroResults = []
    for _numerical_feature in NumericalFeatures:
        _transformed_values = stats.boxcox(SleepDataset[_numerical_feature])[0]
        _shapiro_result = stats.shapiro(_transformed_values)
        _DataShapiroResults.append((_numerical_feature,_shapiro_result.pvalue))

    mo.vstack(
        [
            mo.md("**P-Values of Shapiro-Wilk Tests for the Features After Box-Cox Transformation**"),
            pd.DataFrame(_DataShapiroResults,columns=['Numerical Feature','P-Value']).map(src.OutputFormatting),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 3.2. Categorical Features
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Most of the patients are nurses, doctors or engineers, whose jobs or occupations involve high levels of stress, and most of them have a normal BMI and no sleep disorders. After applying Chi Square test, it can be seen that there are dependent relationships between the categorical features, therefore the use of these features will be more deliberate, as the results could be insignificant.

    Daily stress, time for physical activity, time for personal and recreational activities, diet, and rest time are factors that are dependent on subject's daily routine and lifestyle. By showing that there is evidence of dependence between the categories (features), this premise and relationship can be reinforced.

    And there is no a good representation of categories in `Occupation` nor `BMI Category`, hence some combinations of these values will be more rare to analyze or to learn in a model.
    """)
    return


@app.cell
def _(CategoricalFeatures, SleepDataset):
    mo.vstack(
        [
            mo.md("**Statistics of Categorical Features**"),
            SleepDataset[CategoricalFeatures].describe().iloc[1:],
        ], 
    )
    return


@app.cell
def _(CategoricalFeatures, SleepDataset):
    _fig , _axes = src.CreatePlot(
        2,2,
        (6,6),
    )

    for _ax , _feature in zip(_axes.ravel(),CategoricalFeatures):
        sns.countplot(
            SleepDataset,
            x = _feature,
            ax = _ax,
            color = src.BaseColor,
        )
        _xtick_labels = _ax.get_xticklabels()
        _ax.set_xticks(
            range(len(_xtick_labels)),
            labels=_xtick_labels,
            rotation=90,
        )
        src.SetLabelsToPlot(
            _ax,
            '',
            _feature,
            _ax.get_ylabel(),
            LabelSize = 10,
            TickSize = 9,
        )

    src.SetFigureTitle(_fig,'Distribution of Categorical Features')

    # _fig.savefig(f'../Resources/UnivariatePlot_Categorical.jpg')
    _fig
    return


@app.cell
def _(CategoricalFeatures, NumericalFeatures, SleepDataset):

    _DataChi2Results = []
    for _categorical_feature_1 , _categorical_feature_2 in combinations(CategoricalFeatures,2):
        _chi2_result = stats.chi2_contingency(
            SleepDataset.pivot_table(NumericalFeatures[0],_categorical_feature_1,_categorical_feature_2,'count',fill_value=0)
        )
        _DataChi2Results.append((_categorical_feature_1,_categorical_feature_2,_chi2_result.pvalue))

    mo.vstack(
        [
            mo.md("**P-Values of Chi-Square Tests for the Independence between Features**"),
            pd.DataFrame(_DataChi2Results,columns=['Categorical Feature 1','Categorical Feature 2','P-Value']).map(src.OutputFormatting),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 3.3 Final Observations
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    From Spearman Correlation tests, it can be shown that `Quality of Sleep` of a patient is influenced by factors such as `Physical Activity Level`, `Stress Levels`, and `Sleep Duration`, where the first two relate to lifestyle and quality of life of a subject; additionally, the `Sleep Duration` has a significant influence and impact in determining how well one sleeps. Thus, it follows that some of the factors that measure lifestyle of a subject determine `Quality of Sleep`.

    As assumed, `Occupation` (job) and `BMI Category` of a subject can alter both the quality and duration of sleep, but also the own quality of life of a patient; this dual influence or relationship makes them high-impact factors on overall well-being of a subject and, specifically, how they sleep.

    Based on these observations, a model could be created to evaluate or predict `Quality of Sleep` of a patient based on their lifestyle or quality of life (measured through these factors and features).
    """)
    return


@app.cell
def _(SleepDataset):
    _CorrelationColumns = ['Quality of Sleep','Sleep Duration']
    _CorrelationIndex = ['Physical Activity Level','Stress Level','Daily Steps','Sleep Duration']
    _CorrelationTests = pd.DataFrame(
        columns = _CorrelationColumns,
        index = _CorrelationIndex,
    )

    for _index_label , _column_label in product(_CorrelationIndex,_CorrelationColumns):
        _CorrelationTests.loc[_index_label,_column_label] = stats.spearmanr(
            SleepDataset[_index_label],
            SleepDataset[_column_label],
        ).pvalue

    mo.vstack(
        [
            mo.md("**P-Values of Spearman Correlation Tests**"),
            _CorrelationTests.map(src.OutputFormatting),
        ]
    )
    return


@app.cell
def _(SleepDataset):
    _ANOVAColumns = ['Quality of Sleep','Sleep Duration']
    _ANOVAIndex = ['Occupation','BMI Category']
    _ANOVATests = pd.DataFrame(
        columns = _ANOVAColumns,
        index = _ANOVAIndex,
    )

    for _index_label , _column_label in product(_ANOVAIndex,_ANOVAColumns):
        _CategoriesDataset = SleepDataset.groupby(_index_label)
        _ANOVATests.loc[_index_label,_column_label] = stats.kruskal(
            *[_CategoriesDataset.get_group(_group)[_column_label] for _group in _CategoriesDataset.groups]
        ).pvalue

    mo.vstack(
        [
            mo.md("**P-Values of Kruskal-Wallis Tests**"),
            _ANOVATests.map(src.OutputFormatting),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Multivariate Analysis
    """)
    return


@app.cell
def _():
    mo.md(r"""
    In this section is performed a multivariate analysis to understand how the different features interact with each other using bivariate plots and PCA to support some of previous insights and generate other ones based on the current knowledge.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 4.1. Bivariate Analysis
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    By varying the different numerical and categorical features, it can be shown how each group defined by categorical values behaves differently in terms of numerical variables (specifically `Quality of Sleep` and `Sleep Duration`). This reflects how lifestyle and daily routine interact with patient's quality of life.

    It can be observed `Occupation` of a subject influences their `Quality of Sleep` and `Stress Level`, in addition to the fact that these factors have a negative correlation. The above can be verified in reality by considering that job position directly impacts the stress and pressure someone experiences.

    Obesity and overweight are two conditions that increase the occurrence of conditions such as sleep apnea due to airway obstruction, which can be observed in how `Quality of Sleep` is diminished according to subject's `BMI Category`, as well as the tendency to have more `Sleep Disorder` as weight increases.

    `Sleep Duration` and `Physical Activity Level` have a positive correlation which means that exist a tendency in people to have more physical activity when they have more time to rest (they probably feel more energetic). Which improves their `Quality of Sleep` and reduces their `Strees Level`.

    The values of `BMI Category` follows a irregular distribution on `Occupation` and this is explained by how the data was collected, which will affect the model learning. This is a limitation that should be taken into account.
    """)
    return


@app.cell
def _(CategoricalFeatures, NumericalFeatures):
    # Creating selectors of numerical and categorical features for 
    # box plots of numerical values using categorical values 

    NumericalFeatureOptions_NumCat = mo.ui.dropdown(
        NumericalFeatures,
        value = NumericalFeatures[0],
        label = 'Select a Numerical Feature',
    )

    CategoricalFeatureOptions_NumCat = mo.ui.dropdown(
        CategoricalFeatures,
        value = CategoricalFeatures[0],
        label = 'Select a Categorical Feature',
    )
    return CategoricalFeatureOptions_NumCat, NumericalFeatureOptions_NumCat


@app.cell
def _(
    CategoricalFeatureOptions_NumCat,
    NumericalFeatureOptions_NumCat,
    SleepDataset,
):
    _fig , _axes = src.CreatePlot(FigSize=(5,4))

    _categorical_feature = CategoricalFeatureOptions_NumCat.value
    _numerical_feature = NumericalFeatureOptions_NumCat.value

    _OrderCriteria = SleepDataset.groupby(
        _categorical_feature
    )[_numerical_feature].mean().sort_values().index

    sns.boxplot(
        SleepDataset,
        x = _numerical_feature,
        y = _categorical_feature,
        ax = _axes,
        legend = False,
        color = src.BaseColor,
        order = _OrderCriteria,
    )

    src.SetLabelsToPlot(
        _axes,
        f"{_numerical_feature} vs {_categorical_feature}",
        TitleSize = 14,
        LabelSize = 12,
        TickSize = 10,
    )

    # _fig.savefig(f'../Resources/BivariatePlot_{_categorical_feature.replace(' ','')}_{_numerical_feature.replace(' ','')}.jpg',bbox_inches='tight')
    mo.vstack(
        [
            mo.hstack([NumericalFeatureOptions_NumCat,CategoricalFeatureOptions_NumCat]),
            _fig,
        ]
    )
    return


@app.cell
def _(NumericalFeatures):
    # Creating selectors of numerical features for regression
    # plots of numerical values

    NumericalFeatureOptions_1_NumNum = mo.ui.dropdown(
        NumericalFeatures,
        value = NumericalFeatures[0],
        label = 'Select a Numerical Feature',
    )

    NumericalFeatureOptions_2_NumNum = mo.ui.dropdown(
        NumericalFeatures,
        value = NumericalFeatures[0],
        label = 'Select a Numerical Feature',
    )
    return NumericalFeatureOptions_1_NumNum, NumericalFeatureOptions_2_NumNum


@app.cell
def _(
    NumericalFeatureOptions_1_NumNum,
    NumericalFeatureOptions_2_NumNum,
    SleepDataset,
):
    _fig , _axes = src.CreatePlot(FigSize=(4,4))

    _numerical_feature_1 = NumericalFeatureOptions_1_NumNum.value
    _numerical_feature_2 = NumericalFeatureOptions_2_NumNum.value

    sns.regplot(
        SleepDataset,
        x = _numerical_feature_1,
        y = _numerical_feature_2,
        ax = _axes,
        color = src.BaseColor,
        ci = None,
    )

    src.SetLabelsToPlot(
        _axes,
        f"{_numerical_feature_1} vs {_numerical_feature_2}",
        TitleSize = 14,
        LabelSize = 12,
        TickSize = 10,
    )

    # _fig.savefig(f'../Resources/BivariatePlot_{_numerical_feature_1.replace(' ','')}_{_numerical_feature_2.replace(' ','')}.jpg')
    mo.vstack(
        [
            mo.hstack([NumericalFeatureOptions_1_NumNum,NumericalFeatureOptions_2_NumNum]),
            _fig,
        ]
    )
    return


@app.cell
def _(CategoricalFeatures):
    # Creating selectors of categorical features for pivot
    # tables of categorical values

    CategoricalFeatureOptions_1_CatCat = mo.ui.dropdown(
        CategoricalFeatures,
        value = CategoricalFeatures[0],
        label = 'Select a Categorical Feature',
    )

    CategoricalFeatureOptions_2_CatCat = mo.ui.dropdown(
        CategoricalFeatures,
        value = CategoricalFeatures[1],
        label = 'Select a Categorical Feature',
    )
    return (
        CategoricalFeatureOptions_1_CatCat,
        CategoricalFeatureOptions_2_CatCat,
    )


@app.cell
def _(
    CategoricalFeatureOptions_1_CatCat,
    CategoricalFeatureOptions_2_CatCat,
    NumericalFeatures,
    SleepDataset,
):
    _categorical_feature_1 = CategoricalFeatureOptions_1_CatCat.value
    _categorical_feature_2 = CategoricalFeatureOptions_2_CatCat.value
    try:
        _SummaryCategoricalValues = SleepDataset.pivot_table(
            NumericalFeatures[0],
            _categorical_feature_1,
            _categorical_feature_2,
            'count',
            fill_value = 0,
        )
    except:
        _SummaryCategoricalValues = pd.DataFrame()

    mo.vstack(
        [
            mo.hstack([CategoricalFeatureOptions_1_CatCat,CategoricalFeatureOptions_2_CatCat]),
            mo.md("**Summary of Categorical Values**"),
            _SummaryCategoricalValues,
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 4.2. Principal Component Analysis
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    PCA  is applied on the numerical features to perform a dimensionality reduction, hence using the elbow method on the eigenvalues, the best selection for the number of principal component is three that explains the $87.14\%$ of the variance in the dataset (this percentage is an acceptable reduction of the whole dataset to a few variables).

    Based on an analysis of the loadings, each component explains the next relations and observations in the dataset:

    * *Sleep quality* (PC 1): Its positives loadings are concentrated on how well and long someone sleeps and its negatives on their stress levels. This component highlights how stress negatively impacts recovery efficiency

    * *General health* (PC 2): Its positives loadings explain the increase in blood pressure (hypertension or another conditions) with aging, as well as an increase in exercise to counteract the damage associated with old age. This component illustrates how human susceptibility to disease increases with age

    * *Physical condition* (PC 3): Its positives loadings are related to how much physical activities and exercise someone does and its negatives loadings to the medical condition of their body. This component demonstrates an essential clinical factor for long-term wellness, namely regular physical activity
    """)
    return


@app.cell
def _(NumericalFeatures, SleepDataset):
    PipelinePCA = Pipeline(
        [
            ('Standardization',StandardScaler()),
            ('PrincipalComponents',PCA(random_state=src.RANDOM_STATE))
        ]
    )
    SleepDatasetReducedPCA = PipelinePCA.fit_transform(SleepDataset[NumericalFeatures])

    _fig , _axes = src.CreatePlot(FigSize=(5,4))

    sns.lineplot(
        x = np.arange(1,len(NumericalFeatures)+1),
        y = PipelinePCA['PrincipalComponents'].explained_variance_ratio_.cumsum(),

        color = src.BaseColor,
        linestyle = '--',
        linewidth = 1.5,
        marker = 'o',
        markersize = 6,

        ax = _axes,
    )
    src.SetLabelsToPlot(
        _axes,
        'Scree Plot for Selection of\nNumber of Principal Components',
        'Number of Principal Components',
        'Cumulative Explained Variance',
        TitleSize = 13,
        LabelSize = 11,
        TickSize = 9,
    )

    # _fig.savefig(f'../Resources/PCA_ScreePlotEigen.jpg')
    _fig
    return PipelinePCA, SleepDatasetReducedPCA


@app.cell
def _(NumericalFeatures, PipelinePCA):
    _DataFrameLoadings = pd.DataFrame(
        PipelinePCA['PrincipalComponents'].components_[:3].T,
        columns = [f'PC {_IndexPC}' for _IndexPC in range(1,4)],
        index = NumericalFeatures,
    ).rename_axis('Features')

    mo.vstack(
        [
            mo.md('**Loadings of Features in Each Principal Component**'),
            _DataFrameLoadings.map(partial(src.OutputFormatting,Precision=8)),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Considering that aspects related to physical and sleep health are integrally related, it becomes natural to expect certain patterns when plotting the principal components using `BMI Category` and `Sleep Disorder` as categorical values. Specifically, it can be observed that having a high positive value in PC1 (better sleep health) and a low negative value in PC2 (greater youth) tends to result in normal weight and absence of sleep disorders. And this relationship illustrates the standard optimal health status (young, normal weight and no diseases).

    There is not a clearly separation or grouping between instances, therefore using only numerical features is insufficient to create clusters or including categorical features may be necessary to separate instances.
    """)
    return


@app.cell
def _(CategoricalFeatures, NumericalFeatures):
    # Creating selectos for categorical and numerical features for PCA plots

    CategoricalFeatureOptions_PCA = mo.ui.dropdown(
        CategoricalFeatures,
        value = CategoricalFeatures[0],
        label = 'Select a Categorical Feature',
    )
    NumericalFeatureOptions_PCA = mo.ui.dropdown(
        NumericalFeatures,
        value = NumericalFeatures[0],
        label = 'Select a Numerical Feature',
    )
    return CategoricalFeatureOptions_PCA, NumericalFeatureOptions_PCA


@app.cell
def _(CategoricalFeatureOptions_PCA, SleepDataset, SleepDatasetReducedPCA):
    _fig , _axes = plt.subplot_mosaic(
        '1122\n.33.',
        subplot_kw = {'frame_on':False},
        layout = 'constrained',
        figsize = (6,5),
    )

    _CategoricalFeature = CategoricalFeatureOptions_PCA.value
    _LenCategories = len(SleepDataset[_CategoricalFeature].unique())
    _legend_handles = None
    for (_pc_x , _pc_y) , _index_ax in zip(combinations(range(3),2),range(1,4)):
        _ax = _axes[str(_index_ax)]
        sns.scatterplot(
            x = SleepDatasetReducedPCA[:,_pc_x],
            y = SleepDatasetReducedPCA[:,_pc_y],
            hue = SleepDataset[_CategoricalFeature],

            palette = src.BasePalette(n_colors=_LenCategories),
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
        f'PCA by {_CategoricalFeature}',
        13,
    )

    _fig.legend(
        *_legend_handles,
        title = _CategoricalFeature,
        loc = 'lower right'
    )

    # _fig.savefig(f'../Resources/PCAPlot_{_CategoricalFeature.replace(' ','')}.jpg')
    mo.vstack(
        [
            CategoricalFeatureOptions_PCA,
            _fig
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Using PCA with numerical attributes allows capturing and summarizing the general aspects to describe the subject's profile, achieved through their simple interactions. This can be observed by assigning point sizes according to their `Quality of Sleep` value, which allows seeing in which regions certain values are more concentrated. Specifically, using PC1 and PC2 creates diagonal strips where as their values increases, so does `Quality of Sleep` value along these strips.
    """)
    return


@app.cell
def _(NumericalFeatureOptions_PCA, SleepDataset, SleepDatasetReducedPCA):
    _fig , _axes = plt.subplot_mosaic(
        '1122\n.33.',
        subplot_kw = {'frame_on':False},
        layout = 'constrained',
        figsize = (6,5),
    )

    _NumericalFeature = NumericalFeatureOptions_PCA.value
    _legend_handles = None
    for (_pc_x , _pc_y) , _index_ax in zip(combinations(range(3),2),range(1,4)):
        _ax = _axes[str(_index_ax)]
        sns.scatterplot(
            x = SleepDatasetReducedPCA[:,_pc_x],
            y = SleepDatasetReducedPCA[:,_pc_y],
            size = SleepDataset[_NumericalFeature],

            color = src.BaseColor,
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
        'PCA by Numerical Values',
        13,
    )
    _fig.legend(
        *_legend_handles,
        loc = 'lower right',
        title = _NumericalFeature,
    )

    mo.vstack(
        [
            NumericalFeatureOptions_PCA,
            _fig
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Regression Analysis
    """)
    return


@app.cell
def _():
    mo.md(r"""
    In this section is performed the regression analysis, focusing on model building and assumptions validation assuming the target (`Quality of Sleep`) is continuous. The correlation matrix is examined to show the linear relationships between the variables.
    """)
    return


@app.cell
def _(NumericalFeatures):
    # Splitting numerical features into regressors and target variables

    TargetVariable = NumericalFeatures[2]
    RegressorVariables = NumericalFeatures[:2] + NumericalFeatures[3:]
    return RegressorVariables, TargetVariable


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 5.1. Correlation Matrix
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The correlation matrix shows some evidence of multicollinearity, therefore it is necessary to define models based on selection algorithms (like stepwise selection) to reduce the impact of multicollinearity on the results and predictions. And also there is a correlation between regressor variables and target, making it possible to train a liner models to predict `Quality of Sleep` using only numerical features.

    Through the correlation matrix, it can better appreciate how the different factors that constitute the lifestyle and quality of life of a patient interact to determine how well they sleep. Also noting that some features do not have a significant correlation with the target (`Quality of Sleep`), yet there is an indirect influence; such as `Blood Pressure` values that are correlated with `Age` and `Heart Rate`, and these features have a stronger influence on the `Quality of Sleep` of a person.
    """)
    return


@app.cell
def _(RegressorVariables, SleepDataset, TargetVariable):
    _fig , _axes = src.CreatePlot(FigSize=(6,5))

    _CorrelationValue = SleepDataset[RegressorVariables+[TargetVariable]].corr()
    _MaskValues = np.abs(_CorrelationValue) >= 0.2
    sns.heatmap(
        _CorrelationValue[_MaskValues],
        vmin = -1,
        vmax = 1,
        annot = True,
        annot_kws = {'size':7},
        ax = _axes,
        cmap = src.ColorMapContrast, 
    )
    src.SetLabelsToPlot(
        _axes,
        'Correlation Matrix of Numerical Features',
        TitleSize = 11,
        TickSize = 9, 
    )
    _axes._colorbars[0].tick_params(labelsize=8)

    # _fig.savefig(f'../Resources/CorrelationMatrix.jpg',bbox_inches='tight')
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 5.2. Full Linear Model
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Using only numerical features and a full model shows that all the features are significant, except `Physical Activity Level`, and the regression itself is also significant, this means that the features could be used as a measure of `Quality of Sleep` of a patient. But for the above mentioned some features are collinear, therefore they could be removed to improve the final quality of the model without losing clinical information about a patient while it is reduced the overfit introduced by the additional, redundant (collinear) features.
    """)
    return


@app.cell
def _(NumericalFeatures, RegressorVariables, SleepDataset, TargetVariable):
    LinearModel = smf.ols(
        f"Q('{TargetVariable}') ~ " + ' + '.join([f"Q('{regressor_variable}')" for regressor_variable in RegressorVariables]),
        SleepDataset[NumericalFeatures],
    ).fit()

    LinearModel.summary()
    return (LinearModel,)


@app.cell
def _():
    mo.md(r"""
    This model yields an adjusted R-squared of 0.923, suggesting a near-perfect fit. However, when comparing the predictions through the plot, significant deviations from the true values are observed, particularly for `Quality of Sleep` levels 1, 2, and 6. This discrepancy may indicate model overfitting, potentially caused by variables that are only relevant to specific levels and introduce 'noise' that hides relevant underlying relationships.
    """)
    return


@app.cell
def _(LinearModel, SleepDataset, TargetVariable):
    _fig = src.PlotObservedPredictedValues(SleepDataset[TargetVariable],LinearModel.fittedvalues,'Full')

    # _fig.savefig(f'../Resources/RegressionQuality_FullModel.jpg')
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 5.3. Best Linear Model
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Using Akaike Information Criterion (AIC) for selecting the best suitable subset of features with stepwise algorithm, it can be found that the best model uses only two features and achieves a significative $AIC$ and $F$ scores. This means that this model is slightly better than the full model but not best respect to $R^2_{adj}$ score, although using less features is more suitable to avoid higher variance values and artificial overfit, this means generating better predictions (more interpretable).

    The selected features (`Sleep Duration` and `Stress Level`) align with what empirically measures how well one sleeps, where the subject's stress encompasses their mood, physical condition, and health, while sleep duration determines the feeling of recovery and rest. Therefore, a selection of attributes is obtained that, in a general way, encompasses all aspects of a patient and their sleep quality without exposing sensitive personal information.
    """)
    return


@app.cell
def _(RegressorVariables, SleepDataset, TargetVariable):
    # Using Stepwise Algoithm for selection of the best subset of features

    _LinearModel = LinearRegression()
    _StepwiseAlgorithm = SequentialFeatureSelector(
        _LinearModel,
        k_features = 'best',
        floating = True,
        scoring = src.AkaikeInformationCriterionScore,
        cv = 2,
    )

    _StepwiseAlgorithm.fit(
        SleepDataset[RegressorVariables],
        SleepDataset[TargetVariable]
    )

    BestSubsetRegressionFeatures = [RegressorVariables[_index] for _index in _StepwiseAlgorithm.k_feature_idx_]
    return (BestSubsetRegressionFeatures,)


@app.cell
def _(
    BestSubsetRegressionFeatures,
    NumericalFeatures,
    SleepDataset,
    TargetVariable,
):
    BestLinearModel = smf.ols(
        f"Q('{TargetVariable}') ~ " + ' + '.join([f"Q('{regressor_variable}')" for regressor_variable in BestSubsetRegressionFeatures]),
        SleepDataset[NumericalFeatures],
    ).fit()

    BestLinearModel.summary()
    return (BestLinearModel,)


@app.cell
def _():
    mo.md(r"""
    This model yields an adjusted R-squared of 0.877, representing an optimal fit, although it performs worse than the full model. This is a minor trade-off considering that using fewer attributes increases model interpretability and reduces the artificial overfitting induced by multicollinear variables. The plot shows that the predictions deviate further from the observed values compared to the full model's plot, indicating that the reduced model produces slightly less accurate predictions.
    """)
    return


@app.cell
def _(BestLinearModel, SleepDataset, TargetVariable):
    _fig = src.PlotObservedPredictedValues(SleepDataset[TargetVariable],BestLinearModel.fittedvalues,'Best')

    # _fig.savefig(f'../Resources/RegressionQuality_BestModel.jpg')
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 5.4. Validation of Assumptions
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Due to the non-normal distribution of the features, many of the assumptions inherent in a linear regression model are not satisfied. Key assumptions include normality in residuals and homoscedasticity of the target. These assumptions are verified with the best model defined in the previous section.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 5.4.1. Normality in Residuals
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Since none of the features are normal, their linear combination generates non-normal distributions. And this implies that the residuals of the linear model do not follow a normal distribution and the p-value of Shapiro-Wilk test is 0. Therefore, this assumption is not verified by the model.
    """)
    return


@app.cell
def _(BestLinearModel):
    _QuantilesTheoObs , _RegressionLineParams = stats.probplot(
        BestLinearModel.resid,
        dist = 'norm',
        sparams = (0,BestLinearModel.resid.std(ddof=1))
    )

    _fig , _axes = src.CreatePlot(FigSize=(5,4))

    sns.scatterplot(
        x = _QuantilesTheoObs[0],
        y = _QuantilesTheoObs[1],
        color = src.BaseColor,
        ax = _axes,
    )

    _axes.axline(
        (0,0),
        slope = _RegressionLineParams[0],
        color = 'gray',
        linestyle = ':',
    )

    src.SetLabelsToPlot(
        _axes,
        'Normality of the Residuals',
        'Theorical Quartiles',
        'Residuals Quartiles',
        13,11,9
    )

    # _fig.savefig(f'../Resources/Regression_NormalityResiduals')
    _fig
    return


@app.cell
def _(BestLinearModel):
    _TestResult = stats.shapiro(
        BestLinearModel.resid,
    )

    mo.md(f"**P-Values of Shapiro-Wilk Test: **{_TestResult.pvalue:.6f}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 5.4.2. Homoscedasticity
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    As shown in the plot, the model residuals follow a systematic (functional) pattern and the p-value of Breusch-Pagan test is 0, therefore, they are not homoscedastic and the model predictions will not be robust. Furthermore, the importance and statistical significance of the variables are lower when determining a subject's `Quality of Sleep`.
    """)
    return


@app.cell
def _(BestLinearModel, TargetVariable):
    _fig , _axes = src.CreatePlot(FigSize=(5,4))

    sns.scatterplot(
        x = BestLinearModel.fittedvalues,
        y = BestLinearModel.resid,
        color = src.BaseColor,
        ax = _axes,
    )
    src.SetLabelsToPlot(
        _axes,
        'Residuals as a Function of Predicted Values',
        TargetVariable,
        'Residuals',
        13,11,9,
    )

    # _fig.savefig(f'../Resources/Regression_PredictedResiduals')
    _fig
    return


@app.cell
def _(BestLinearModel):
    _TestResult = het_breuschpagan(
        BestLinearModel.resid,
        BestLinearModel.model.exog
    )[1]

    mo.md(f"**P-Values of Breusch-Pagan Test: **{_TestResult:.6f}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Factor Analysis
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Factor analysis is performed with numerical features and encoded categorical variables to encompass all possible interactions between features that can be explained through the factors.

    Using the elbow method on the eigenvalues of the factors, it is determined that using four factors allows explaining $89.31\%$ of the data variance (only positive eigenvalues are considered). Additionally, it can be noted that negative eigenvalues were found, which implies the existence of highly correlated features (multicollinearity), a fact that can be observed in the correlation matrix and in the auxiliary plots using categorical features.
    """)
    return


@app.cell
def _(SleepDataset):
    # Encoding categorical features of the dataset and 
    # applying standard scaler over all the features

    SleepDataset_Processed = SleepDataset.copy(True)

    for _ordinal_feature , _categories in zip(['Gender','BMI Category'],[['Female','Male'],['Normal','Overweight','Obese']]):
        _OrdinalEncoder = OrdinalEncoder(categories=[_categories])
        SleepDataset_Processed[_ordinal_feature] = _OrdinalEncoder.fit_transform(SleepDataset_Processed[[_ordinal_feature]])

    for _one_hot_feature in ['Occupation','Sleep Disorder']:
        _OneHotEncoder = OneHotEncoder(sparse_output=False)
        _OneHotValues = _OneHotEncoder.fit_transform(SleepDataset_Processed[[_one_hot_feature]])
        SleepDataset_Processed.drop(columns=_one_hot_feature,inplace=True)
        SleepDataset_Processed[_one_hot_feature + ' :: ' + np.array(*_OneHotEncoder.categories_)] = _OneHotValues

    # Saving the final preprocessed dataset

    try:
        SleepDataset_Processed.to_csv(
            src.PATH + 'ProcessedSleepDataset.csv'
        )
    except:
        pass

    _NumericalScaler = StandardScaler()
    SleepDataset_Processed[SleepDataset_Processed.columns] = _NumericalScaler.fit_transform(SleepDataset_Processed)
    return (SleepDataset_Processed,)


@app.cell
def _(SleepDataset_Processed):
    _FactorAnalysis = Factor(
        SleepDataset_Processed,
    )
    _FAResults = _FactorAnalysis.fit()
    _Eigenvalues = _FAResults.eigenvals

    _PositiveEigenvalues = _Eigenvalues > 0
    _ExplainedVariance = _Eigenvalues[_PositiveEigenvalues]
    _ExplainedVariance_Ratio = (_ExplainedVariance/_ExplainedVariance.sum()).cumsum()

    _fig , _axes = src.CreatePlot(FigSize=(5,4))

    sns.lineplot(
        x = range(1,len(_ExplainedVariance)+1),
        y = _ExplainedVariance_Ratio,

        color = src.BaseColor,
        linestyle = '--',
        linewidth = 1.5,
        marker = 'o',
        markersize = 6,

        ax = _axes,
    )
    src.SetLabelsToPlot(
        _axes,
        'Scree Plot for Selection of\nNumber of Factors',
        'Number of Factors',
        'Explained Variance',
        13,11,9
    )

    # _fig.savefig(f'../Resources/FactorAnalysis_ScreePlot.jpg')
    _fig
    return


@app.cell
def _(SleepDataset_Processed):
    # Performing the factor analysis with 4 factors 
    # based on the observations

    FactorAnalysis = Factor(
        SleepDataset_Processed,
        n_factor = 4, 
    )

    FactorAnalysisResults = FactorAnalysis.fit()
    return (FactorAnalysisResults,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Using the mean of the communalities, it is found that the Factor Analysis model has moderate quality, meaning that not all variables are adequately explained by the factors. Although some of the variances of the variables are adequately explained by the factors (such as `Gender`, `Quality of Sleep`, `Blood Pressure`, `Sleep Disorder`, `BMI Category`, `Age`, `Physical Activity`), some others are not (such as `Occupation`, `Daily Steps`).

    The above compromises the confidence in the interpretation of the factors and in the final quality of the results, due to insufficient data representation in `Occupation` (this implies that some occupations cannot be adequately explained with the other features). Therefore, the loadings of the variables that are adequately explained by the factors have greater importance in what each factor represents for the data. Based on this, the factors could be a good low-dimensional representation of the dataset that summarizes it.
    """)
    return


@app.cell
def _(FactorAnalysisResults, SleepDataset_Processed):
    _RelevantCommunalities = FactorAnalysisResults.communality >= 0
    _SortedFilteredCommunalities = sorted(
        zip(SleepDataset_Processed.columns[_RelevantCommunalities],FactorAnalysisResults.communality[_RelevantCommunalities]),
        key = lambda communality : communality[1],
        reverse = True,
    )
    _MeanCommunality = FactorAnalysisResults.communality.mean()

    _fig , _axes = src.CreatePlot(FigSize=(6,6.5))

    sns.barplot(
        x = [_communality[0] for _communality in _SortedFilteredCommunalities],
        y = [_communality[1] for _communality in _SortedFilteredCommunalities],
        color = src.BaseColor,
        ax = _axes,
    )
    _axes.text(12,1.2,f'Mean Communality: {_MeanCommunality:.4f}',size=11,horizontalalignment='center')

    src.SetLabelsToPlot(
        _axes,
        'Quality of Communalities',
        'Features',
        'Communality',
        13,11,9
    )
    _axes.tick_params(axis='x',rotation=90)

    # _fig.savefig(f'../Resources/FactorAnalysis_Communalities.jpg')
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 6.1. Factors Interpretation
    """)
    return


@app.cell
def _():
    mo.md(r"""
    * *Factor 1*: Pertains to explaining the health of a patient (their precarity or deficiencies) based on their physical condition and sleep disorders. This factor includes features like: `BMI Category`, `Blood Preasure`, `Sleep Disorder`.

    * *Factor 2*: Is associated with the overall quality of sleep, how well one sleeps and recovers, also encompassing how having a stressful life affects sleep (high stress levels and hypertension). This factor includes features like: `Sleep Duration`, `Quality of Sleep`, `Strees Level`.

    * *Factor 3*: Is linked to the physical activity and activation levels of a patient and their connection to the presence of insomnia (possible relationship between the energy and mood someone has throughout the day). This feature includes: `Physical Activity Level`, `Daily Steps`.

    * *Factor 4*: Does not provide relevant information or relationships (Only explains the gender of a patient).
    """)
    return


@app.cell
def _(FactorAnalysisResults, SleepDataset_Processed):
    _fig , _axes = plt.subplots()

    sns.heatmap(
        FactorAnalysisResults.loadings,
        cmap = src.ColorMapContrast,
        vmax = 1,
        vmin = -1,
        annot = True,
        annot_kws = {'size':7.5},
        ax = _axes
    )
    _axes.set_xticklabels(range(1,5))
    _axes.set_yticklabels(SleepDataset_Processed.columns)

    src.SetLabelsToPlot(
        _axes,
        'Factor Loadings',
        'Factors',
        'Features',
        13,11,8
    )
    _axes._colorbars[0].tick_params(labelsize=8)

    # _fig.savefig(f'../Resources/FactorAnalysis_FactorLoadings.jpg',bbox_inches='tight')
    _fig
    return


@app.cell
def _(FactorAnalysisResults, SleepDataset_Processed):
    _SummaryFactorsLoadings = [
        mo.md('**Relevant Features in each Factor**'),
        mo.md('A feature is considred relevant if its loading is greater than 0.4.')
    ]

    _Loadings = FactorAnalysisResults.loadings
    _DataframesFeaturesByFactors = dict()
    for _factor in range(4):
        _significant_loadings = np.abs(_Loadings[:,_factor]) > 0.4

        _features_loadings = []
        for _feature , _load in zip(SleepDataset_Processed.columns[_significant_loadings],_Loadings[_significant_loadings,_factor]):
            _features_loadings.append([_feature,f'{_load:.4f}'])

        _factor_eigenvalue = f'*Factor {_factor+1}. Eigenvalue: {FactorAnalysisResults.eigenvals[_factor]:.6f}*'
        _DataframesFeaturesByFactors[_factor_eigenvalue] = pd.DataFrame(_features_loadings,columns=['Feature','Loading'])

    for _factor , _dataframe_loadings in _DataframesFeaturesByFactors.items():
        _SummaryFactorsLoadings.extend([mo.md(_factor),_dataframe_loadings])

    mo.vstack(_SummaryFactorsLoadings)
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
