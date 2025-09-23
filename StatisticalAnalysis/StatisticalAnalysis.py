import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")

with app.setup:
    # Import auxiliar libraries
    import marimo as mo
    from itertools import combinations


    # Importing libraries
    import pandas as pd
    import numpy as np

    import seaborn as sns
    import matplotlib.pyplot as plt

    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    from scipy import stats


    # Importing Functions and Utils
    import SourceStatisticalAnalysis as src


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
    SleepDataset_Raw.groupby(
        'Sleep Disorder',
        dropna = False,
    )['Gender'].count()
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
        None of the features are normal, so some of the techniques that will be used will lead to insignificant results. Therefore, the values could be transformed with power transformations like Box-Cox or it could be assumed that the results will be insignificant.
    
        After using Box-Cox transformation there was no improve (the transformed distributions were still non-normal under Shapiro-Wilk test), therefore the analysis of the results using the techniques that will be used will be more detailed and thorough.
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
        Most of the patients are nurses, doctors or engineers, whose jobs or occupations involve high levels of stress, and most of them have a normal BMI and no sleep disorders.
    
        After applying Chi Square test, it can be seen that there are dependent relationships between the categorical features, therefore the use of this features will be more deliberate, as the results could be insignificant.
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
    mo.md(r"## 4. Multivariate Exploratory")
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
    mo.md(r"### 5.2. Full Linear Mode")
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


if __name__ == "__main__":
    app.run()
