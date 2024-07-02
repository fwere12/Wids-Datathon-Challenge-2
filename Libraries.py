import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading in both the datasets
test_data = pd.read_csv("C:/Users/Administrator/Downloads/widsdatathon2024-challenge2/test.csv")
train_data = pd.read_csv("C:/Users/Administrator/Downloads/widsdatathon2024-challenge2/train.csv")
test_data.head(5)
train_data.head(5)

#Get the shape of the dataframe
shape_train = train_data.shape
print("Shape of train data:", shape_train)

test_shape = test_data.shape
print("Shape of test data:", test_shape)