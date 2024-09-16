import pandas as pd
import pickle


def transform_feature(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Apply quadratic transformation to a specified feature.

    :param df: The DataFrame containing the data.
    :param feature: The feature to transform.
    :return: DataFrame with the transformed feature.
    """
    df[f'{feature}_squared'] = df[feature] ** 2
    return df


def load_model_with_pickle(filepath: str):
    """
    Load the saved model from a file using pickle.

    :param filepath: The path where the model is saved.
    :return: The loaded model.
    """
    with open(filepath, 'rb') as file:
        return pickle.load(file)


def save_model_with_pickle(model, filepath: str) -> None:
    """
    Save the trained model to a file using pickle.

    :param model: The trained model to be saved.
    :param filepath: The path where the model will be saved.
    """
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)