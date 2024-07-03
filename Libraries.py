import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading in both the datasets
test_data = pd.read_csv("C:/Users/Administrator/Downloads/widsdatathon2024-challenge2/test.csv")
train_data = pd.read_csv("C:/Users/Administrator/Downloads/widsdatathon2024-challenge2/train.csv")
test_data.head(5)
train_data.head(5)

#Chhecking for our data shape
test_shape = test_data.shape
print(test_shape)

train_shape = train_data.shape
print(train_shape)

#Checking for missing values in our datasets
test_data.isnull().sum()
train_data.isnull().sum()

#Summary statistics
test_data.describe().T
train_data.describe().T