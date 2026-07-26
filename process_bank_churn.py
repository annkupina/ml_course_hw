from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def remove_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that are not used for model training.

    Args:
        df: Input dataframe.

    Returns:
        A copy of the dataframe without unused columns.
    """
    return df.drop(columns=["Surname"])


def split_data(
    df: pd.DataFrame,
    target_col: str = "Exited",
    test_size: float = 0.25,
    random_state: int = 35,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and validation sets.

    Args:
        df: Dataset.
        target_col: Name of the target column.
        test_size: Fraction of validation data.
        random_state: Random seed.

    Returns:
        Training and validation dataframes.
    """
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )


def split_features_and_target(
    df: pd.DataFrame,
    input_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target.

    Args:
        df: Input dataframe.
        input_cols: Feature columns.
        target_col: Target column.

    Returns:
        Features and target.
    """
    return df[input_cols].copy(), df[target_col].copy()


def get_column_types(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Identify numeric and categorical columns.

    Args:
        df: Feature dataframe.

    Returns:
        Numeric columns and categorical columns.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    return numeric_cols, categorical_cols


def encode_features(
    df: pd.DataFrame,
    encoder: OneHotEncoder,
    categorical_cols: list[str],
) -> pd.DataFrame:
    """
    Apply one-hot encoding to categorical features.

    Args:
        df: Feature dataframe.
        encoder: Fitted encoder.
        categorical_cols: Categorical columns.

    Returns:
        Dataframe with encoded categorical features.
    """
    encoded = encoder.transform(df[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index,
    )

    df = df.drop(columns=categorical_cols)

    return pd.concat([df, encoded_df], axis=1)


def scale_features(
    df: pd.DataFrame,
    scaler: Optional[MinMaxScaler],
    numeric_cols: list[str],
) -> pd.DataFrame:
    """
    Scale numeric features.

    Args:
        df: Feature dataframe.
        scaler: Fitted scaler.
        numeric_cols: Numeric columns.

    Returns:
        Dataframe with scaled numeric features.
    """
    if scaler is not None:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


def transform_features(
    df: pd.DataFrame,
    encoder: OneHotEncoder,
    scaler: Optional[MinMaxScaler],
) -> pd.DataFrame:
    """
    Apply all preprocessing transformations to a feature dataframe.

    Args:
        df: Feature dataframe.
        encoder: Fitted OneHotEncoder.
        scaler: Optional fitted MinMaxScaler.

    Returns:
        Transformed feature dataframe.
    """
    numeric_cols, categorical_cols = get_column_types(df)

    df = encode_features(df, encoder, categorical_cols)
    df = scale_features(df, scaler, numeric_cols)

    return df


def preprocess_data(
    raw_df: pd.DataFrame,
    scaler_numeric: bool = False,
):
    """
    Train preprocessing objects and transform train/validation datasets.

    Args:
        raw_df: Raw dataset.
        scaler_numeric: Whether to scale numeric features.

    Returns:
        X_train,
        y_train,
        X_val,
        y_val,
        input_cols,
        scaler,
        encoder
    """
    df = remove_unused_columns(raw_df)

    train_df, val_df = split_data(df)

    input_cols = list(train_df.columns)[2:-1]
    target_col = "Exited"

    X_train, y_train = split_features_and_target(
        train_df, input_cols, target_col
    )
    X_val, y_val = split_features_and_target(
        val_df, input_cols, target_col
    )

    numeric_cols, categorical_cols = get_column_types(X_train)

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
    ).fit(X_train[categorical_cols])

    scaler = None
    if scaler_numeric:
        scaler = MinMaxScaler().fit(X_train[numeric_cols])

    X_train = transform_features(X_train, encoder, scaler)
    X_val = transform_features(X_val, encoder, scaler)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        input_cols,
        scaler,
        encoder,
    )


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: list[str],
    scaler: Optional[MinMaxScaler],
    encoder: OneHotEncoder,
) -> pd.DataFrame:
    """
    Preprocess new unseen data using fitted preprocessing objects.

    Args:
        new_df: New data.
        input_cols: Training feature columns.
        scaler: Fitted scaler.
        encoder: Fitted encoder.

    Returns:
        Preprocessed dataframe.
    """
    inputs = new_df[input_cols].copy()
    return transform_features(inputs, encoder, scaler)
