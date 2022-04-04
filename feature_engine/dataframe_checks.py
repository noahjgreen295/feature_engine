"""Series of checks to be performed on dataframes used as inputs of methods fit() and
transform().
"""

from typing import Any, List, Union

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from feature_engine.numpy_to_pandas import _is_numpy, _numpy_to_dataframe, _numpy_to_series


def _is_dataframe(X: Union[np.ndarray, np.generic, pd.DataFrame]) -> pd.DataFrame:
    """
    Checks if the input is a DataFrame and then creates a copy.

    If the input is a numpy array, it converts it to a pandas Dataframe. This is mostly
    so that we can add the check_estimator checks for compatibility with sklearn.

    In addition, allows Feature-engine transformers to be used within a Scikit-learn
    Pipeline together with Scikit-learn transformers like the SimpleImputer, which
    return by default Numpy arrays.

    Parameters
    ----------
    X : pandas Dataframe or numpy array. The one that will be checked and copied.

    Raises
    ------
    TypeError
        If the input is not a Pandas DataFrame or a numpy array.

    Returns
    -------
    X : pandas Dataframe.
        A copy of original DataFrame. Important step not to accidentally transform the
        original dataset entered by the user.
    """
    # check_estimator uses numpy arrays for its checks.
    # Thus, we need to allow np arrays.
    if _is_numpy(X):
        X = _numpy_to_dataframe(X)

    _check_if_input_is_sparse_empty_or_not_df(X)

    return X.copy()


def _check_if_input_is_sparse_empty_or_not_df(X: Any) -> None:
    """
    Checks if input is a sparse, not a DataFrame or an empty DataFrame.

    Parameters
    ----------
    X : the object to test.

    Raises
    ------
    TypeError
        If the input is not a Pandas DataFrame, and empty dataframe or a sparse array.
    """
    if issparse(X):
        raise TypeError("This transformer does not support sparse matrices.")

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X is not a pandas dataframe. The dataset should be a pandas dataframe."
        )

    if X.empty:
        raise ValueError(
            "0 feature(s) (shape=%s) while a minimum of %d is "
            "required." % (X.shape, 1)
        )


def _check_input_matches_training_df(X: pd.DataFrame, reference: int) -> None:
    """
    Checks that DataFrame to transform has the same number of columns that the
    DataFrame used with the fit() method.

    Parameters
    ----------
    X : Pandas DataFrame
        The df to be checked.
    reference : int
        The number of columns in the dataframe that was used with the fit() method.

    Raises
    ------
    ValueError
        If the number of columns does not match.

    Returns
    -------
    None
    """

    if X.shape[1] != reference:
        raise ValueError(
            "The number of columns in this dataset is different from the one used to "
            "fit this transformer (when using the fit() method)."
        )

    return None


def _check_contains_na(X: pd.DataFrame, variables: List[Union[str, int]]) -> None:
    """
    Checks if DataFrame contains null values in the selected columns.

    Parameters
    ----------
    X : Pandas DataFrame
    variables : List
        The selected group of variables in which null values will be examined.

    Raises
    ------
    ValueError
        If the variable(s) contain null values
    """

    if X[variables].isnull().values.any():
        raise ValueError(
            "Some of the variables to transform contain NaN. Check and "
            "remove those before using this transformer."
        )


def _check_contains_inf(X: pd.DataFrame, variables: List[Union[str, int]]) -> None:
    """
    Checks if DataFrame contains inf values in the selected columns.

    Parameters
    ----------
    X : Pandas DataFrame
    variables : List
        The selected group of variables in which null values will be examined.

    Raises
    ------
    ValueError
        If the variable(s) contain np.inf values
    """

    if np.isinf(X[variables]).values.any():
        raise ValueError(
            "Some of the variables to transform contain inf values. Check and "
            "remove those before using this transformer."
        )


def _check_pd_X_y(
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
):
    """
    Ensures X and y are compatible pandas DataFrame and Series. If both are pandas
    objects, checks their indexes match. If any is a numpy array, converts to pandas
    object with compatible index.

    Parameters
    ----------
    X: Pandas DataFrame or numpy ndarray
    y: Pandas Series or numpy ndarray

    Raises
    ------
    ValueError: if X and y are pandas objects with inconsistent indexes.
    TypeError: if X is sparse matrix, empty dataframe or not a dataframe.
    TypeError: if y can't be parsed as pandas Series.

    Returns
    -------
    X: Pandas DataFrame
    y: Pandas Series
    """

    # Check X
    if _is_numpy(X):
        X = _numpy_to_dataframe(X, index=y.index if isinstance(y, pd.Series) else None)

    _check_if_input_is_sparse_empty_or_not_df(X)

    # Check y
    if _is_numpy(y):
        y = _numpy_to_series(y, index=X.index)  # X is a df at this point

    elif not isinstance(y, pd.Series):
        TypeError(f"y must be a pandas Series or numpy array. Got {y} instead.")

    # If both parameters are pandas objects, and their indexes are inconsistent,
    # an exception is raised (i.e. this is the caller's error.)
    if len(X) != len(y) or not all(y.index == X.index):
        raise ValueError("The indexes of X and y do not match.")
    else:
        X = X.copy()
        y = y.copy()

    return X, y
