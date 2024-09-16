import pandas as pd

from utils import transform_feature, load_model_with_pickle


def main():
    """
    Main function to load the model, perform inference, and display predictions.
    """

    # Load the model
    model = load_model_with_pickle('linear_regression_model.pkl')

    # Load inference data
    df_infer = pd.read_csv('hidden_test.csv')

    # Apply the same quadratic transformation
    df_infer = transform_feature(df_infer, '6')

    # Extract features for inference
    X_infer = df_infer[['6_squared']]

    # Make predictions
    predictions = model.predict(X_infer)

    # Create a DataFrame with the predictions
    results = pd.DataFrame({'Predictions': predictions})

    # Save predictions to a CSV file
    results.to_csv('predictions.csv', index=False)

    print("Predictions saved to 'predictions.csv'.")


if __name__ == '__main__':
    main()
