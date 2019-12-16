# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 23:51:46 2019
@author: OphirShurany
"""
#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.close('all')

#%% create dataframe
df = pd.read_csv("bank.csv",sep='|',encoding='utf8');
df=df.drop('Unnamed: 0',axis=1)
#view first 5 rows in df
df.head()
#presenting all columns, number of rows and type
df.info()
#feature statistics for numerical categories
df.describe()

#FOR 6.1 =============================================================================
# sns.countplot(x='y',data=df)
# d1=df.copy()
# d2=d1[d1.y=='yes']
# d3=d1[d1.y=='no']
# while len(d3.y)>=len(d2.y):
#     d1=pd.concat([d1, d2])
#     d2=d1[d1.y=='yes']
# df=d1
# sns.countplot(x='y',data=df)
# =============================================================================
#%%
#change "yes" or "no" to 1 or 0
df['y'] = df.y.map(dict(yes=1, no=0))
header = ['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
df.hist(column=header,figsize=(10,10))
plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
plt.show()
#%%
# Convert the month list to 4 binary quarters column 
months=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'];
Q = [1,1,1,1,2,2,2,3,3,3,4,4,4];month_dic=dict(zip(months,Q))
df['month']=df.month.replace(month_dic)
df=pd.get_dummies(df, columns=['month'],prefix='Q')
df['education']=df.education.replace(['basic.6y','basic.4y', 'basic.9y'], 'basic')
for col in list(df.columns):
    plt.figure()
    sns.countplot(x=col,hue='y',data=df)
#%% Convert other Series from yes or no to binary
df['default'] = df.default.map(dict(yes=1, no=0))
df['housing'] = df.housing.map(dict(yes=1, no=0))
df['loan'] = df.loan.map(dict(yes=1, no=0))
#%%Convert Categorial to numeric
df['day_of_week']=df.day_of_week.astype('category').cat.codes
df['contact']=df.contact.astype('category').cat.codes
df['poutcome']=df.poutcome.astype('category').cat.codes
#Convert Categorial to numeric and remains NaN
df['education'] = df.education.astype('category').cat.codes
df.education.replace({-1: np.nan}, inplace=True)
df['marital']=df.marital.astype('category').cat.codes
df.marital.replace({-1: np.nan}, inplace=True)
df.job.replace({"unknown": np.nan}, inplace=True)
df['job']=df.job.astype('category').cat.codes
df.job.replace({-1: np.nan}, inplace=True)
#The significant Variables are 'education', 'job', 'housing', and 'loan'.
sns.countplot(x='education',hue='y',data=df)
#%%correlation heat map
plt.figure()
cor = df.corr()
cor.head()
sns.heatmap(cor, annot=False,cmap='coolwarm')
#%%Missing Values
#show null 
print(df.isna().sum())
#BOX PLOT
for col in list(df.columns):
    plt.figure()
    sns.boxplot(x='y', y=col, data=df)
#%%
from sklearn.preprocessing import MinMaxScaler, StandardScaler    
idx_numeric=[0,10,11,12,14,15,16,17,18]
scaler = MinMaxScaler()
df[df.columns[idx_numeric]] = scaler.fit_transform(df[df.columns[idx_numeric]])
#%%
from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=3, min_samples=2).fit(df)