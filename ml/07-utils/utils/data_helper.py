import pandas as pd
from typing import Optional, Union

def load_data(
    filepath: str,
    index_col: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """Load a dataset from a file (CSV, Excel, etc.) with optional arguments.

    Args:
        filepath: Path to the data file.
        index_col: Column to set as index.
        **kwargs: Additional arguments for pd.read_csv/read_excel.

    Returns:
        pd.DataFrame: Loaded data.
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath, index_col=index_col, **kwargs)
    elif filepath.endswith(('.xlsx', '.xls')):
        return pd.read_excel(filepath, index_col=index_col, **kwargs)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """Split data into features and target, then into train/test sets.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        test_size: Fraction of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
