import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")

with app.setup:
    # Import auxiliar libraries
    import marimo as mo


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
    
        * `Gender`
        * `Age`
        * `Occupation`
        * `Sleep Duration`
        * `Quality of Sleep`
        * `Physical Activity Level`
        * `Stress Level`
        * `BMI Category`
        * `Blood Pressure`
        * `Heart Rate`
        * `Daily Steps`
        * `Sleep Disorder`
    
        There are missing values on `Sleep Disorder` because of there are patients without sleep disorders.  The notation of `Blood Pressure` is Systolic/Diastolic form which will be transformed.
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
    mo.md(r"The missing values of `Sleep Disorder` are imputed with `No` and the values of `Blood Pressure` are splited into systolic and diastolic values.")
    return


@app.cell
def _(SleepDataset_Raw):
    SleepDataset = SleepDataset_Raw.copy()

    # Filling missing values

    SleepDataset['Sleep Disorder'] = SleepDataset['Sleep Disorder'].fillna('No')

    # Splitting blood pressure into systolic and diastolic

    SleepDataset[['Blood Pressure Systolic','Blood Pressure Diastolic']] = [*SleepDataset['Blood Pressure'].apply(src.SplitBloodPressure)]
    SleepDataset.drop(columns=['Blood Pressure'],inplace=True)
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
def _():
    mo.md(r"")
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
        None of the features are normal, so some of the techniques that will be used will yield insignificant results. Therefore, the values could be transformed with power transformations like Box-Cox or it could be assumed that the results will be insignificant.
    
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
    )

    for _ax , _feature in zip(_axes.ravel(),NumericalFeatures):
        KindPlotNumericalFeatures.value(
            SleepDataset,
            x = _feature,
            ax = _ax,
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
    mo.md(r"")
    return


@app.cell
def _(CategoricalFeatures, SleepDataset):
    _fig , _axes = plt.subplots(
        2,2,
        figsize = (9,9),
        layout = 'constrained',
        gridspec_kw={'wspace':0.1,'hspace':0.1},
    )

    for _ax , _feature in zip(_axes.ravel(),CategoricalFeatures):
        sns.countplot(
            SleepDataset,
            x = _feature,
            ax = _ax,
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


if __name__ == "__main__":
    app.run()
