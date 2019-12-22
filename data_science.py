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
df = pd.read_csv("bank.csv",sep='|');
df_copy=df.copy()
#drop duplicate data
df = df.drop_duplicates('Unnamed: 0',keep=False)
#drop # and duration
df=df.drop(['Unnamed: 0','duration'],axis=1)
#%%view first 5 rows in df
df.head()
#presenting all columns, number of rows and type
df.info()
#feature statistics for numerical categories
df.describe()
#%% Histograms for categorial features
cat_features = df.select_dtypes(include="object").columns
df_group=df.groupby("y")
df_group_yes=df_group.get_group("yes")
df_group_no=df_group.get_group("no")
for feature in cat_features:
    feature_df = pd.DataFrame()
    feature_df["no"] = df_group_no[feature].value_counts()
    feature_df["yes"] = df_group_yes[feature].value_counts()
    #plt.figure()
    feature_df.plot(kind='bar')
    plt.title("Feature Histogram - " + feature)
    plt.ylabel("Count")
    plt.xlabel("Feature unique values")
    plt.tight_layout()
#%% Histograms for Numeric features
num_features = list(df.select_dtypes(exclude="object").columns)
#sns.pairplot(df,vars=num_features,hue="y")
for feature in num_features:
    plt.figure()
    df_group=df.groupby("y")
    df_group_yes=df_group.get_group("yes")
    df_group_no=df_group.get_group("no")
    #plt.figure()
    plt.hist([df_group_yes[feature],df_group_no[feature]],label=["yes","no"])
    #plt.hist(df_group_yes[feature],label=feature)
    plt.legend()
    plt.title("Feature Histogram - " + feature)
    plt.ylabel("Count")
    plt.xlabel("Feature unique values")
    plt.tight_layout()
#%%
#change "yes" or "no" to 1 or 0
df['y'] = df.y.map(dict(yes=1, no=0))
#%% Convert the month list to 4 binary quarters column 
months=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'];
Q = [1,1,1,1,2,2,2,3,3,3,4,4,4];month_dic=dict(zip(months,Q))
df['month']=df.month.replace(month_dic)
df=pd.get_dummies(df, columns=['month'],prefix='Q')
#convert all basics to 1 basic
df['education']=df.education.replace(['basic.6y','basic.4y', 'basic.9y'], 'basic')
education_dic={'illiterate': 0,'basic' : 1,'high.school' : 2,'professional.course' : 3,'university.degree' : 4}
df['education']=df.education.replace(education_dic)
df['job']=df.job.replace('entrepreneur', 'self-employed')
df.job.replace(['retired', 'unemployed'], 'no_active_income', inplace=True)
df.job.replace(['services', 'housemaid'], 'services', inplace=True)
df.job.replace(['admin.', 'management'], 'administration_management', inplace=True)
df.marital.replace('divorced', 'single', inplace=True)
#%% Convert other Series from yes or no to binary
df['default'] = df.default.map(dict(yes=1, no=0))
df['housing'] = df.housing.map(dict(yes=1, no=0))
df['loan'] = df.loan.map(dict(yes=1, no=0))
df=df.rename(columns = {'contact':'contact_by_cellular'})
df['contact_by_cellular'] = df.contact_by_cellular.map(dict(cellular = 1, telephone = 0))
df['poutcome'] = df.poutcome.map(dict(success = 1, nonexistent = 0,failure=-1))
#%% Pdays manipulations
df = df.drop(['pdays','default'], axis=1);
#%%Missing Values
#show null
df = df.replace('unknown',np.nan) 
df.isna().sum()
print("total NaN rows = " + str(sum(df.isna().sum())))
#Categorial features
#job
df.loc[(df['job'].isnull()==True) & (df['education']==1), 'job'] = 'blue-collar'
df.loc[(df['age'] > 30 ) & (df['job'] == 'administration_management' ) , 'job'] = 'retired'
df = df[pd.notnull(df['job'])]
#education 
df.loc[(df['education'].isnull()==True) & (df['job']=='administration_management'), 'education'] = 4
df.loc[(df['education'].isnull()==True) & (df['job']=='services'), 'education'] = 2
#Numeric features: age, campaign, default, loan
df = df.fillna(df.mean())
df = df.dropna()
print("Now the total NaN rows = " + str(sum(df.isna().sum())))
#%%correlation heat map
plt.figure(figsize=(16,16))
# Separate both dataframes into 
numeric_df = df.select_dtypes(exclude="object")
# categorical_df = df.select_dtypes(include="object")
cor = numeric_df.corr()
plt.title("Correlation Matrix", fontsize=16)
sns.heatmap(cor, annot=False,square = True,cbar=True,cmap='coolwarm')
#%%#BOX PLOT
# for col in numeric_df.columns:
#     plt.figure()
#     sns.boxplot(x='y', y=col, hue="y",data=numeric_df)
#%%
from sklearn.preprocessing import MinMaxScaler   
from sklearn.preprocessing import Normalizer
#separate the data from the target attributes
scaler = MinMaxScaler((-1,1))
normalized_df_data =scaler.fit_transform(numeric_df.values)
X=numeric_df
X=X.dropna()
 # NOTE=============================================================================
# Outliers: Outliers are defined as 1.5 x Q3 value (75th percentile).
# From the above table, it can be seen that only 'age' and 'campaign'
# have outliers as max('age') and max('campaign') > 1.5Q3('age') and >1.5Q3('campaign') respectively.
# But we also see that the value of these outliers are not so unrealistic
# (max('age')=98 and max('campaign')=56).
# Hence, we need not remove them since the prediction model 
# should represent the real world. This improves the 
# generalizability of the model and makes it robust 
# for real world situations. 
# The outliers, therefore, are not removed.
# =============================================================================
#%%The optimal value for epsilon will be found at the point of maximum curvature.
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
#%%
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=1.2, min_samples=50).fit(X)
labels=db.labels_
clusterNum=len(set(labels))
#%%
df_copy["cluster_Db"]=labels
realClusterNum=len(set(labels))-(1 if -1 in labels else 0)
clusterNum=len(set(labels))
#%%


