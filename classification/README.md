# MNIST classifier
This folder contains solution for MNIST classifier task in `mnist_classifier.py`, which consists of:
1. `DigitClassificationInterface`: interface for models.
2. `CNNModel`: class for CNN model, which implements `DigitClassificationInterface`.
3. `RandomForestModel`: class for Random Forest model, which implements `DigitClassificationInterface`.
4. `RandomModel`: class for Random model, which implements `DigitClassificationInterface`.
5. `DigitClassifier`: class for classifier, which uses models to classify digits. It takes as an input parameter the name of the algorithm
   and provides predictions with exactly the same structure (inputs and outputs) not
   depending on the selected algorithm.

# Setup
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

# Example of usage
```
import numpy as np

# Generate sample input data
 sample_image_28x28x1 = np.random.rand(28, 28, 1)  # Random 28x28x1 image

 # Initialize classifiers
 cnn_classifier = DigitClassifier('cnn')
 rf_classifier = DigitClassifier('rf')
 rand_classifier = DigitClassifier('rand')

 # Test CNNModel
 print("Testing CNNModel:")
 cnn_prediction = cnn_classifier.predict(sample_image_28x28x1)
 print(f"Predicted digit (CNN): {cnn_prediction}")

 # Test RandomForestModel
 print("Testing RandomForestModel:")
 # Note: RandomForestModel requires a trained model, this is just a placeholder
 # You should train the model before using it
 rf_prediction = rf_classifier.predict(sample_image_28x28x1)
 print(f"Predicted digit (RandomForest): {rf_prediction}")

 # Test RandomModel
 print("Testing RandomModel:")
 rand_prediction = rand_classifier.predict(sample_image_28x28x1)
 print(f"Predicted digit (RandomModel): {rand_prediction}")
```