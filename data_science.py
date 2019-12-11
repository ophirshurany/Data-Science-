# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 23:51:46 2019
@author: OphirShurany
"""
#import packages
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%% create dataframe
df = pd.read_csv("bank.csv",sep='|',encoding='utf8');
#view first 5 rows in df
df.head()
#presenting all columns, number of rows and type
df.info()
#feature statistics for numerical categories
df.describe()
#change "yes" or "no" to 1 or 0
df['y'] = df.y.eq('yes').mul(1)
# Convert the month list to 4 binary quarters column 
months=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'];
df['Q1']=df.month.replace(months,[1,1,1,0,0,0,0,0,0,0,0,0]);
df['Q2']=df.month.replace(months,[0,0,0,1,1,1,0,0,0,0,0,0]);
df['Q3']=df.month.replace(months,[0,0,0,0,0,0,1,1,1,0,0,0]);
df['Q4']=df.month.replace(months,[0,0,0,0,0,0,0,0,0,1,1,1]);
df=df.drop('month', axis=1);
#correlation heat map
plt.figure()
cor = df.corr()
sns.heatmap(cor, annot=True, cmap='jet')
plt.show()
#%%Missing Values
#show null 
df.isnull().sum()