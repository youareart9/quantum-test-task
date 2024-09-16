# Regression task
This folder contains solution for regression task and consists of:
1. `exploratory_data_analysis.ipynb`: Jupyter notebook with exploratory data analysis.
2. `train.py`: Python script for Linear Regression model training.
3. `predict.py`: Python script for model inference on test data.
4. `utils.py`: Python script with utility functions.
5. `requirements.txt`: File with required Python packages.
6. `predictions.csv`: File with predictions on `hidden_test.csv`.

# Setup
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

# Example of usage
1. Train model. To be able to train model, you need to have `train.csv` in the same folder. Script will save trained model in `linear_regression_model.pkl`.
```
python3 train.py
```
2. Inference on test data. To be able to make predictions, you need to have `hidden_test.csv` in the same folder and trained model `linear_regression_model.pkl`. Script will save predictions in `predictions.csv`.
```
python3 predict.py
```
