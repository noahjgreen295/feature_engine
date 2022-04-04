"""Functions to detect numpy objects and convert to pandas objects."""

from typing import Any, List, Union

import numpy as np
import pandas as pd


def _is_numpy(obj_in: Any) -> bool:
    """
    Checks if object is a numpy object.

    Parameters
    ----------
    obj_in : the object to test.

    Returns
    -------
    True if object is a numpy object, else False
    """
    return isinstance(obj_in, (np.generic, np.ndarray))


def _numpy_to_dataframe(
        obj_in: Union[np.generic, np.ndarray],
        index: Union[np.generic, np.ndarray, pd.Series, None] = None
) -> pd.DataFrame:
    """
    Converts a numpy object to a pandas DataFrame.

    Parameters
    ----------
    obj_in : the object to convert.
    index : array-like (optional); will set index on DataFrame.

    Returns
    -------
    df_out : the object converted to a pandas DataFrame.
    """
    col_names: List[str] = [str(i) for i in range(obj_in.shape[1])]
    df_out: pd.DataFrame = pd.DataFrame(obj_in, columns=col_names, index=index)

    return df_out


def _numpy_to_series(
        obj_in: Union[np.generic, np.ndarray],
        index: Union[np.generic, np.ndarray, pd.Series, None] = None,
) -> pd.Series:
    """
    Converts a numpy object to a pandas Series.

    Parameters
    ----------
    obj_in : the object to convert.
    index : array-like (optional); will set index on Series.

    Returns
    -------
    df_out : the object converted to a pandas Series
    """
    s_out: pd.Series = pd.Series(obj_in, index=index)

    return s_out
