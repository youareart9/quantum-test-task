import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
import numpy as np

from utils import transform_feature, save_model_with_pickle


def perform_cross_validation(X: pd.DataFrame, y: pd.Series, model, n_splits: int = 5) -> np.ndarray:
    """
    Perform K-fold cross-validation and return RMSE scores.

    :param X: The feature set.
    :param y: The target variable.
    :param model: The model to evaluate.
    :param n_splits: Number of folds for cross-validation.
    :return: An array of RMSE scores for each fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    neg_mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    return np.sqrt(-neg_mse_scores)


def main():
    """
    Main function to load data, transform features, perform cross-validation, and train the model.
    """
    # Load dataset
    df = pd.read_csv('train.csv')

    # Quadratic transformation of feature '6'
    df = transform_feature(df, '6')

    # Extract features and target
    X = df[['6_squared']]
    y = df['target']

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Perform cross-validation and calculate RMSE
    rmse_cv_scores = perform_cross_validation(X, y, model)

    # Train the model on the entire dataset
    model.fit(X, y)

    # Display metrics
    print(f"Cross-Validation RMSE Scores: {rmse_cv_scores}")
    print(f"Mean CV RMSE: {rmse_cv_scores.mean()}")

    # Save the trained model using pickle
    save_model_with_pickle(model, 'linear_regression_model.pkl')
    print("Model saved to 'linear_regression_model.pkl'.")


if __name__ == '__main__':
    main()
