import os

import pandas as pd 
import numpy as np

class KNNClassifier:
    def __init__(self, k: int) -> None:
        self.k = k

    def fit(self, x: np.array, y: np.array):
        self.x = x
        self.y = y

    def predict(self, x: np.array):
        pass
