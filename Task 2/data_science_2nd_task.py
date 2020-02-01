"""
Created on Fri Jan 31 18:22:18 2020
@author: Ophir Shurany
"""
#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.utils import shuffle
import random
import warnings
warnings.filterwarnings('ignore')
plt.close('all')
import time
start_time = time.time()
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
#%% Create dataframe
data = pd.read_csv("pulsar_stars.csv")
#view first 5 rows in df
data.head()
#presenting all columns, number of rows and type
data.info()
#feature statistics for numerical categories
data.describe()
#Majority class is 0 (Not a pulsar)
print("Total NaN rows = " + str(sum(data.isna().sum())))
#%%
cols = list(data.columns)
features = cols
features.remove('target_class')
X=data[features]
Y=data.target_class
X_train_test, X_val, Y_train_test, Y_val = model_selection.train_test_split(X, Y, test_size=0.1, random_state=42)

