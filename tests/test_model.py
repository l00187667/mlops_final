import unittest
import pickle
import numpy as np

class TestCO2Model(unittest.TestCase):
    def test_prediction(self):
        model = pickle.load(open("model/model.pkl", "rb"))
        result = model.predict(np.array([[1200]]))
        self.assertTrue(result[0] > 0)

if __name__ == '__main__':
    unittest.main()
