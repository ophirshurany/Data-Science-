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
df=df.drop(['Unnamed: 0','duration'],axis=1)
#view first 5 rows in df
df.head()
#presenting all columns, number of rows and type
df.info()
#feature statistics for numerical categories
df.describe()
#%%
#change "yes" or "no" to 1 or 0
#df['y'] = df.y.map(dict(yes=1, no=0))

#%%
# Convert the month list to 4 binary quarters column 
months=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'];
Q = [1,1,1,1,2,2,2,3,3,3,4,4,4];month_dic=dict(zip(months,Q))
df['month']=df.month.replace(month_dic)
df=pd.get_dummies(df, columns=['month'],prefix='Q')
df['Q_1']=df.Q_1.replace([1,0],["yes","no"])
df['Q_2']=df.Q_2.replace([1,0],["yes","no"])
df['Q_3']=df.Q_3.replace([1,0],["yes","no"])
df['Q_4']=df.Q_4.replace([1,0],["yes","no"])
df['education']=df.education.replace(['basic.6y','basic.4y', 'basic.9y'], 'basic')
#%%
categorcial_variables = ['job', 'marital', 'education', 'default', 'loan', 'contact', 'poutcome','y', 'Q_1','Q_2','Q_3','Q_4']
freq_pos = (df.y.values == 'yes').sum()
freq_neg = (df.y.values == 'no').sum()
for col in categorcial_variables:
    plt.figure(figsize=(10,4))
    sns.barplot(df[col].value_counts().values, df[col].value_counts().index)
    plt.title(col)
    plt.tight_layout()
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
#%%correlation heat map
plt.figure()
cor = df.corr(method='spearman')
cor.head()
sns.heatmap(cor, annot=False,cmap='coolwarm')
#%%Missing Values
#show null 
print(df.isna().sum())
#missing data @ age,marital,education.default, housing,loan and campain
