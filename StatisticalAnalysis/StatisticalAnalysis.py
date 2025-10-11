import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")

with app.setup:
    # Import auxiliar libraries
    import marimo as mo
    from itertools import combinations , product


    # Importing libraries
    import pandas as pd
    import numpy as np

    import seaborn as sns
    import matplotlib.pyplot as plt

    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    from scipy import stats

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    from mlxtend.feature_selection  import SequentialFeatureSelector


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
    return (RANDOM_STATE,)


@app.cell
def _():
    mo.md(r"Sleep quality may be influenced by habits and lifestyle of a person; therefore, understanding how these values and measures influence a person could determine how well they sleep. Thus, this study examines how various factors that determine lifestyle of a person can impact their sleep quality.")
    return


@app.cell
def _():
    mo.md(r"## 1. Load Dataset and First Exploration")
    return


@app.cell
def _():
    mo.md(
        r"""
        The dataset contains `374` instances, with `12` attributes which describe the sleep health of a patient. These columns are:
    
        * `Gender`: Nominal variable with Male and Female values 
    
        * `Age`: Discrete variable with range of values from 27 to 59 years
    
        * `Occupation`: Nominal variable with Software Engineer, Doctor, Sales Representative, Teacher, Nurse, Engineer, Accountant, Scientist, Lawyer, Salesperson and Manager values
    
        * `Sleep Duration`: Continuous variable with range of values from 5.8 to 8.5 hours
    
        * `Quality of Sleep`: Discrete variable with range of values from 4 to 9
    
        * `Physical Activity Level`: Discrete variable with range of values from 30 to 90
    
        * `Stress Level`: Discrete variable with range of values from 3 to 8
    
        * `BMI Category`: Ordinal variable with Normal, Normal Weight, Overweight and Obese values
    
        * `Blood Pressure`: Ordinal variable for pressure measure as Systolic/Diastolic form
    
        * `Heart Rate`: Discrete variable with range of values from 65 to 86 BPM
    
        * `Daily Steps`: Discrete variable with range of values from 3000 to 10000 steps
    
        * `Sleep Disorder`: Nominal variable with Sleep Apnea and Insomnia values
    
        There are missing values on `Sleep Disorder` because of there are patients without sleep disorders.  The notation of `Blood Pressure` is Systolic/Diastolic form which will be transformed. There is a duplicate category in `BMI Category` (Normal and Normal Weight) which will be removed. `Quality of Sleep` is the feature to predict (target) and study in this analysis.
        """
    )
    return


@app.cell
def _():
    # Defining useful variables

    PATH = './Datasets/'
    PATH_DATASET = PATH + 'Sleep_health_and_lifestyle_dataset.csv'
    return (PATH_DATASET,)


@app.cell
def _(PATH_DATASET):
    # Loading dataset

    SleepDataset_Raw = pd.read_csv(
        PATH_DATASET,
        index_col = 0,
    )
    return (SleepDataset_Raw,)


@app.cell
def _(SleepDataset_Raw):
    mo.vstack(
        [
            mo.md("**Example of Instances**"),
            SleepDataset_Raw,

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


@app.cell
def _():
    mo.md(r"## 2. Data Transformation")
    return


@app.cell
def _():
    mo.md(r"The missing values of `Sleep Disorder` are imputed with `No`, the values of `Blood Pressure` are splitted into systolic and diastolic values, and the values with `Normal Weight` in `BMI Category` are transformed to `Normal`.")
    return


@app.cell
def _(SleepDataset_Raw):
    SleepDataset = SleepDataset_Raw.copy()

    # Filling missing values

    SleepDataset['Sleep Disorder'] = SleepDataset['Sleep Disorder'].fillna('No')

    # Splitting blood pressure into systolic and diastolic

    SleepDataset[['Blood Pressure Systolic','Blood Pressure Diastolic']] = [*SleepDataset['Blood Pressure'].apply(src.SplitBloodPressure)]
    SleepDataset.drop(columns=['Blood Pressure'],inplace=True)

    # Remove duplicate category in BMI Category

    _IndexBMINormalWeight = SleepDataset.query("`BMI Category` == 'Normal Weight'").index
    SleepDataset.loc[_IndexBMINormalWeight,'BMI Category'] = 'Normal'
    return (SleepDataset,)


@app.cell
def _(SleepDataset):
    mo.vstack(
        [
            mo.md("**Example of Instances After Transformations**"),
            SleepDataset,

        ], 
        align = 'center',
    )
    return


@app.cell
def _():
    mo.md(r"## 3. Univariate Analysis")
    return


@app.cell
def _(SleepDataset):
    # Splitting features into numerical and categorical features

    NumericalFeatures , CategoricalFeatures = src.SplitFeatures(SleepDataset)
    return CategoricalFeatures, NumericalFeatures


@app.cell
def _():
    mo.md(r"### 3.1. Numerical Features")
    return


@app.cell
def _():
    mo.md(
        r"""
        None of the features are normal, so some of the techniques that will be used will lead to insignificant results. Therefore, the values could be transformed with power transformations like Box-Cox or it could be assumed that the results will be insignificant. After using Box-Cox transformation there was no improve (the transformed distributions were still non-normal under Shapiro-Wilk test), therefore the analysis of the results using the techniques that will be used will be more detailed and thorough.
    
        Fifty percent of patients have a `Quality of Sleep` between 6 and 8, and a `Sleep Duration` of between 6.4 to 7.8 hours. This can be explained by considering that stress and physical activity influence sleep onset and the recovery of the body during sleep. To this, it can add the biological degradation of the body as a person becomes older, which impacts the number of hours needed to feel rested after sleeping.
        """
    )
    return


@app.cell
def _(NumericalFeatures, SleepDataset):
    mo.vstack(
        [
            mo.md("**Statistics of Numerical Features**"),
            SleepDataset[NumericalFeatures].describe().iloc[1:],
        ], 
    )
    return


@app.cell
def _():
    KindPlotNumericalFeatures = mo.ui.dropdown(
        {'Violin':sns.violinplot,'Box':sns.boxplot,'Histogram':sns.histplot},
        value = 'Box',
        label = 'Choose a Kind of Plot: ',
    )
    return (KindPlotNumericalFeatures,)


@app.cell
def _(KindPlotNumericalFeatures, NumericalFeatures, SleepDataset):
    _fig , _axes = plt.subplots(
        3,3,
        figsize = (12,12),
        layout = 'constrained',
        gridspec_kw={'wspace':0.1,'hspace':0.1},
        subplot_kw = {'frame_on':False},
    )

    for _ax , _feature in zip(_axes.ravel(),NumericalFeatures):
        KindPlotNumericalFeatures.value(
            SleepDataset,
            x = _feature,
            ax = _ax,
            color = src.BaseColor,
        )
        _ax.set_xlabel('')
        _ax.set_title(_feature,size=16)
        _ax.tick_params(axis='both',labelsize=12)
        _ax.set_ylabel(_ax.get_ylabel(),size=14)

    _fig.suptitle('Distribution of Numerical Features',size=24)

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
            pd.DataFrame(_DataShapiroResults,columns=['Numerical Feature','P-Value']),
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
            pd.DataFrame(_DataShapiroResults,columns=['Numerical Feature','P-Value']),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"### 3.2. Categorical Features")
    return


@app.cell
def _():
    mo.md(
        r"""
        Most of the patients are nurses, doctors or engineers, whose jobs or occupations involve high levels of stress, and most of them have a normal BMI and no sleep disorders. After applying chi square test, it can be seen that there are dependent relationships between the categorical features, therefore the use of this features will be more deliberate, as the results could be insignificant. 
    
        Daily stress, time for physical activity, time for personal and recreational activities, diet, and rest time are factors that are subject to a daily routine and lifestyle of a person. By showing that there is evidence of dependence between the categories (features), this premise and relationship can be reinforced.
        """
    )
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
    _fig , _axes = plt.subplots(
        2,2,
        figsize = (9,9),
        layout = 'constrained',
        gridspec_kw={'wspace':0.1,'hspace':0.1},
        subplot_kw = {'frame_on':False},
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
        _ax.set_xlabel('')
        _ax.set_title(_feature,size=16)
        _ax.tick_params(axis='both',labelsize=12)
        _ax.set_ylabel(_ax.get_ylabel(),size=14)

    _fig.suptitle('Distribution of Categorical Features',size=24)

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
            pd.DataFrame(_DataChi2Results,columns=['Categorical Feature 1','Categorical Feature 2','P-Value']),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"### 3.3 Final Observations and Conclusions")
    return


@app.cell
def _():
    mo.md(
        r"""
        From Spearman correlation tests, it can be shown that `Quality of Sleep` of a patient is influenced by factors such as `Physical Activity Level`, `Stress Levels`, and `Sleep Duration`, where the first two relate to lifestyle and quality of life of a person; additionally, the `Sleep Duration` has a significant influence and impact in determining how well one sleeps. Thus, it follows that some of the factors that measure lifestyle of a person determine `Quality of Sleep`.
    
        As assumed, `Occupation` (job) and `BMI Category` of a person can alter both the quality and duration of sleep, but also the own quality of life of a patient; this dual influence or relationship makes them high-impact factors on overall well-being of a person and, specifically, how they sleep.
    
        Based on these observations, a model could be created to evaluate or predict `Quality of Sleep` of a person based on their lifestyle or quality of life (measured through these factors and features).
        """
    )
    return


@app.cell
def _(SleepDataset):
    _CorrelationColumns = ['Quality of Sleep','Sleep Duration']
    _CorrelationIndex = ['Physical Activity Level','Stress Level','Daily Steps','Sleep Duration']
    _CorrelationTests = pd.DataFrame(
        columns = _CorrelationColumns,
        index = _CorrelationIndex,
    )

    for _IndexLabel , _ColumnLabel in product(_CorrelationIndex,_CorrelationColumns):
        _CorrelationTests.loc[_IndexLabel,_ColumnLabel] = stats.spearmanr(
            SleepDataset[_IndexLabel],
            SleepDataset[_ColumnLabel],
        ).pvalue

    mo.vstack(
        [
            mo.md("**P-Values of Spearman Correlation Tests**"),
            _CorrelationTests,
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

    for _IndexLabel , _ColumnLabel in product(_ANOVAIndex,_ANOVAColumns):
        _CategoriesDataset = SleepDataset.groupby(_IndexLabel)
        _ANOVATests.loc[_IndexLabel,_ColumnLabel] = stats.kruskal(
            *[_CategoriesDataset.get_group(_group)[_ColumnLabel] for _group in _CategoriesDataset.groups]
        ).pvalue

    mo.vstack(
        [
            mo.md("**P-Values of Kruskal-Wallis Tests**"),
            _ANOVATests,
        ]
    )
    return


@app.cell
def _():
    mo.md(r"## 4. Multivariate Exploratory")
    return


@app.cell
def _():
    mo.md(r"### 4.1. Bivariate Analysis")
    return


@app.cell
def _():
    mo.md(
        r"""
        By varying the different numerical and categorical features, it can be shown how each group defined by categorical values behaves differently in terms of numerical variables (specifically `Quality of Sleep` and `Sleep Duration`). This reflects how lifestyle and daily routine interact with quality of life of a person.
    
        It can be observed `Occupation` of a person influences their `Quality of Sleep` and `Stress Level`, in addition to the fact that these factors have a negative correlation. The above can be verified in reality by considering that job position directly impacts the stress and pressure someone experiences.
    
        Obesity and overweight are two conditions that increase the occurrence of conditions such as sleep apnea due to airway obstruction, which can be observed in how `Quality of Sleep` is diminished according to `BMI Category` of a person, as well as the tendency to have more `Sleep Disorder` as weight increases.
        """
    )
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
    _fig , _axes = plt.subplots(
        subplot_kw = {'frame_on':False},
    )

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

    _axes.set_title(
        f"{_numerical_feature} vs {_categorical_feature}",
        size = 14,
    )

    # _fig.savefig(f'./Resources/BivariatePlot_{_categorical_feature.replace(' ','')}_{_numerical_feature.replace(' ','')}.jpg')
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
    _fig , _axes = plt.subplots(
        subplot_kw = {'frame_on':False},
    )

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

    _axes.set_title(
        f"{_numerical_feature_1} vs {_numerical_feature_2}",
        size = 14,
    )

    # _fig.savefig(f'./Resources/BivariatePlot_{_numerical_feature_1.replace(' ','')}_{_numerical_feature_2.replace(' ','')}.jpg')
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


@app.cell
def _(NumericalFeatures, RANDOM_STATE, SleepDataset):
    _DimensionalReduction = PCA(
        whiten = True,
        random_state = RANDOM_STATE,
    )
    _NumericalDatasetReduction = _DimensionalReduction.fit_transform(
        SleepDataset[NumericalFeatures]
    )
    # print(_DimensionalReduction.explained_variance_ratio_)
    # print(_DimensionalReduction.components_[1])

    # sns.scatterplot(
        # x = _NumericalDatasetReduction[:,0],
        # y = _NumericalDatasetReduction[:,1],
    # )
    return


@app.cell
def _():
    mo.md(r"### 4.2. Principal Component Analysis")
    return


@app.cell
def _():
    mo.md(
        r"""
        Using elbow method on the eigenvalues, the best selection for the number of principal component is three that explains the $87.14\%$ of the variance in the dataset. Which comes from:
        * *Sleep quality* (PC 1): Its positives loadings are concentrated on how well and long someone sleeps and its negatives on their stress levels
        * *General health* (PC 2): Its positives loadings explain the increase in blood pressure (hypertension or another conditions) and in old age, as well as an increase in exercise to counteract the damage associated with old age
        * *Physical condition* (PC 3): Its positives loadings are related to how much physical activities and exercise someone does and its negatives loadings to the medical condition of their body
        """
    )
    return


@app.cell
def _(NumericalFeatures, RANDOM_STATE, SleepDataset):
    PipelinePCA = Pipeline(
        [
            ('Standardization',StandardScaler()),
            ('PrincipalComponents',PCA(random_state=RANDOM_STATE))
        ]
    )
    SleepDatasetReducedPCA = PipelinePCA.fit_transform(SleepDataset[NumericalFeatures])

    _fig , _axes = plt.subplots(
        subplot_kw = {'frame_on':False},
    )

    sns.lineplot(
        x = np.arange(1,len(NumericalFeatures)+1),
        y = PipelinePCA['PrincipalComponents'].singular_values_,

        color = src.BaseColor,
        linestyle = '--',
        linewidth = 1.5,
        marker = 'o',
        markersize = 6,

        ax = _axes,
    )
    _axes.set_xlabel('Principal Components',size=12)
    _axes.set_ylabel('Eigenvalues',size=12)
    _axes.set_title('Scree Plot for Selection of\nNumber of Principal Components',size=14)
    _axes.tick_params(axis='both',labelsize=10)

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
            mo.md('**Loadings of Each Principal Component**'),
            _DataFrameLoadings,
        ]
    )
    return


@app.cell
def _():
    mo.md(
        r"""
        Considering that aspects related to physical and sleep health are integrally related, it becomes natural to expect certain patterns when plotting the principal components using `BMI Category` and `Sleep Disorder` as categorical values. Specifically, it can be observed that having a high positive value in PC1 (better sleep health) and a low negative value in PC2 (greater youth) tends to result in normal weight and absence of sleep disorders.
    
        There is not a clearly separation or grouping between instances, therefore using only numerical features is insufficient to create clusters or including categorical features may be necessary to separate instances.
        """
    )
    return


@app.cell
def _(CategoricalFeatures, NumericalFeatures):
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
        figsize = (7,5),
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

        _ax.set_xlabel(f'PC {_pc_x+1}',size=9)
        _ax.set_ylabel(f'PC {_pc_y+1}',size=9)
        _ax.tick_params(axis='both',labelsize=8)

        _legend_handles = _ax.get_legend_handles_labels()
        _ax.legend_ = None

    _fig.suptitle('PCA by Categorical Values',size=12)
    _fig.legend(
        *_legend_handles,
        title = _CategoricalFeature,
        loc = 'lower right'
    )

    # _fig.savefig(f'./Resources/PCAPlot_{_CategoricalFeature.replace(' ','')}.jpg')
    mo.vstack(
        [
            CategoricalFeatureOptions_PCA,
            _fig
        ]
    )
    return


@app.cell
def _():
    mo.md(r"Using PCA with numerical attributes allows capturing and summarizing the general aspects to describe the profile of a person, achieved through their simple interactions. This can be observed by assigning point sizes according to their `Quality of Sleep` value, which allows seeing in which regions certain values are more concentrated. Specifically, using PC1 and PC2 creates diagonal strips where as their values increases, so does `Quality of Sleep` value along these strips.")
    return


@app.cell
def _(NumericalFeatureOptions_PCA, SleepDataset, SleepDatasetReducedPCA):
    _fig , _axes = plt.subplot_mosaic(
        '1122\n.33.',
        subplot_kw = {'frame_on':False},
        layout = 'constrained',
        figsize = (7,5),
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

        _ax.set_xlabel(f'PC {_pc_x+1}',size=9)
        _ax.set_ylabel(f'PC {_pc_y+1}',size=9)
        _ax.tick_params(axis='both',labelsize=8)

        _legend_handles = _ax.get_legend_handles_labels()
        _ax.legend_ = None

    _fig.suptitle('PCA by Numerical Values',size=12)
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


@app.cell
def _():
    mo.md(r"## 5. Regression Analysis")
    return


@app.cell
def _(NumericalFeatures):
    # Splitting numerical features into regressors and target variables

    TargetVariable = NumericalFeatures[2]
    RegressorVariables = NumericalFeatures[:2] + NumericalFeatures[3:]
    return RegressorVariables, TargetVariable


@app.cell
def _():
    mo.md(r"### 5.1. Correlation Matrix")
    return


@app.cell
def _():
    mo.md(r"The correlation matrix shows some evidence of multicollinearity, therefore it is necessary to define models based on selection algorithms (as stepwise selection) to reduce the impact of multicollinearity on the results and predictions. And also there is a correlation between regressor variables and target, making it possible to create liner models to predict `Quality of Sleep`.")
    return


@app.cell
def _(RegressorVariables, SleepDataset, TargetVariable):
    _fig , _axes = plt.subplots(
        figsize = (12,9)
    )

    _CorrelationValue = SleepDataset[RegressorVariables+[TargetVariable]].corr()
    _MaskValues = np.abs(_CorrelationValue) >= 0.2
    sns.heatmap(
        _CorrelationValue[_MaskValues],
        vmin = -1,
        vmax = 1,
        annot = True,
        annot_kws = {'size':14},
        ax = _axes,
        cmap = src.ColorMapContrast, 
    )
    _axes.set_title('Correlation Matrix of Numerical Features',size=24)
    _axes.tick_params(axis='both',labelsize=16)

    _fig
    return


@app.cell
def _():
    # _fig , _axes = plt.subplots(
    #     2,2,
    #     figsize = (9,9),
    #     layout = 'constrained',
    #     gridspec_kw={'wspace':0.1,'hspace':0.1},
    #     subplot_kw = {'frame_on':False},
    # )

    # for _ax , _feature in zip(_axes.ravel(),CategoricalFeatures):
    #     sns.boxplot(
    #         SleepDataset,
    #         x = _feature,
    #         y = TargetVariable,
    #         ax = _ax,
    #         color = src.BaseColor,
    #     )
    #     _xtick_labels = _ax.get_xticklabels()
    #     _ax.set_xticks(
    #         range(len(_xtick_labels)),
    #         labels=_xtick_labels,
    #         rotation=90,
    #     )
    #     _ax.set_xlabel('')
    #     _ax.set_title(_feature,size=16)
    #     _ax.tick_params(axis='both',labelsize=12)
    #     _ax.set_ylabel(_ax.get_ylabel(),size=14)

    # _fig.suptitle('Distribution of Categorical Features',size=24)

    # _fig
    return


@app.cell
def _():
    mo.md(r"### 5.2. Full Linear Model")
    return


@app.cell
def _():
    mo.md(r"Using a full model shows that all the features are significant, except `Physical Activity Level`, and the regression itself is also significant, this means that the features could be used as a measure of quality of sleep of a patient. But for the above mentioned some features are collinear, therefore they could be removed to improve the final quality of the model.")
    return


@app.cell
def _(NumericalFeatures, RegressorVariables, SleepDataset, TargetVariable):
    LinearModel = smf.ols(
        f"Q('{TargetVariable}') ~ " + ' + '.join([f"Q('{regressor_variable}')" for regressor_variable in RegressorVariables]),
        SleepDataset[NumericalFeatures],
    ).fit()

    print(LinearModel.summary())
    return


@app.cell
def _():
    mo.md(r"### 5.3. Best Linear Model")
    return


@app.cell
def _():
    mo.md(r"Using Akaike Information Criterion (AIC) for selecting the best suitable subset of features with stepwise algorithm, it can be found that the best model uses only two features and achieves a significative $AIC$ and $F$ scores. This means that this model is slightly better than the full model but not best respect to $R^2_{adj}$ score, although using less features is more suitable to avoid higher variance values and artificial overfit. Therefore this model is better than the full model.")
    return


@app.cell
def _(RegressorVariables, SleepDataset, TargetVariable):
    _LinearModel = LinearRegression()
    _StepwiseAlgorithm = SequentialFeatureSelector(
        _LinearModel,
        k_features = 'best',
        floating = True,
        scoring = src.AkaikeInformationCriterionScore,
        cv = 2,
        n_jobs = -1,
        pre_dispatch = 'all',
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

    print(BestLinearModel.summary())
    return


if __name__ == "__main__":
    app.run()
