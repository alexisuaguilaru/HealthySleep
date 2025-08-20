import marimo

__generated_with = "0.14.17"
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
    mo.md(r"The missing values of `Sleep Disorder` are imputed and the values of `Blood Pressure` are splited into systolic and diastolic values.")
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


if __name__ == "__main__":
    app.run()
