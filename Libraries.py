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

#plot target distribution
train_data['metastatic_diagnosis_period'].plot(kind='hist',title='Target Distribution')

#Checking for missing values in our datasets
test_data.isnull().sum()
train_data.isnull().sum()

#Summary statistics
test_data.describe().T
train_data.describe().T


#Get the shape of the dataframe
shape_train = train_data.shape
print("Shape of train data:", shape_train)

test_shape = test_data.shape
print("Shape of test data:", test_shape)

#Importing the rest of the libraries
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,root_mean_squared_error
from sklearn.model_selection import RepeatedStratifiedKFold,StratifiedKFold,KFold
pd.set_option('display.max_columns', 185)
pd.set_option('display.max_rows',185)
# text preprocessing modules
import re 
from string import punctuation 
import warnings
warnings.filterwarnings("ignore")
# seeding
np.random.seed(123)

y = train_data['metastatic_diagnosis_period'] #target
data = pd.concat((train_data,test_data)).reset_index(drop=True).copy()
data=data.drop(columns=['patient_id','metastatic_diagnosis_period'],axis=1)

#engineer some features
data["clust"]=(data.metastatic_cancer_diagnosis_code.str.len()==4).astype("int")
data["is_female"] = data.breast_cancer_diagnosis_desc.str.contains("female").astype("int")