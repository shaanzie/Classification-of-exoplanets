import pandas as pd
import random
import math

dataset = pd.read_csv('../new_data.csv')]


positiveX = []
negativeX = []

for(i, v) in enumerate dataset:
    if v == 0:
        negativeX.append(X[i])