import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

class TestIrisModel(unittest.TestCase):
    def setUp(self):
        # Load the saved model
        self.model = joblib.load("artifacts/model.joblib")
        # Sample input data for testing (representative of Iris dataset features)
        self.test_data = pd.DataFrame(
            [
                [5.1, 3.5, 1.4, 0.2],  # Likely Iris-setosa
                [6.5, 3.0, 4.5, 1.5],  # Likely Iris-versicolor
                [7.2, 3.2, 6.0, 1.8],  # Likely Iris-virginica
            ],
            columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        )
        # Expected labels for the test data
        self.expected_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    def test_model_prediction(self):
        # Test model predictions
        predictions = self.model.predict(self.test_data)
        # Verify predictions match expected labels
        self.assertEqual(list(predictions), self.expected_labels, "Predictions do not match expected labels")
    
    def test_model_accuracy(self):
        # Test model accuracy on a small sample
        predictions = self.model.predict(self.test_data)
        accuracy = accuracy_score(self.expected_labels, predictions)
        self.assertGreaterEqual(accuracy, 0.9, "Model accuracy is below 90%")

if __name__ == '__main__':
    unittest.main()