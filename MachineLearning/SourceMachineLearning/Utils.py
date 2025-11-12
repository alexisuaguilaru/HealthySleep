import pandas as pd
def SplittingFeatures(
        Dataset: pd.DataFrame,
        Features: list[str],
    ) -> tuple[list[str],list[str],list[str]]:
    """
    Function for splitting features of a 
    dataset based on their data type and 
    number of unique values.

    Parameters
    ----------
    Dataset: pd.DataFrame
        Dataset whose features are being splitted
    Features: list[str]
        Features that are being splitted

    Returns
    -------
    Numerical: list[str]
        Numerical features (integer, float)
    Binary: list[str]
        Binary features (boolean)
    Categorical: list[str]
        Categorical features
    """

    Numerical , Binary , Categorical = [] , [] , []
    for feature in Features:
        if Dataset[feature].dtype == 'object':
            if len(Dataset[feature].unique()) == 2:
                Binary.append(feature)
            else:
                Categorical.append(feature)
        else:
            Numerical.append(feature)

    return Numerical , Binary , Categorical