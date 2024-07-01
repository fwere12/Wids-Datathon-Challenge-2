import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading in both the datasets
test_data = pd.read_csv("C:/Users/Administrator/Downloads/widsdatathon2024-challenge2/test.csv")
train_data = pd.read_csv("C:/Users/Administrator/Downloads/widsdatathon2024-challenge2/train.csv")
test_data.head(5)
train_data.head(5)