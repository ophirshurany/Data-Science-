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
#drop duplicate data
df = df.drop_duplicates('Unnamed: 0',keep=False)
#drop #rows
df=df.drop('Unnamed: 0',axis=1)
#we drop duration as well because high correlation. We save it for later
duration=df["duration"]
df_copy_1=df #Keep original
df=df.drop('duration',axis=1)
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
#%%section 3 - missing data
df = df.replace('unknown',np.nan)
"""check = df.isnull() """
"""df_without_default = df 
del df_without_default['default']
nan_rows_without_def = df
_without_default.isnull().sum(axis = 1).sum() """
nan_rows = df.isnull().sum(axis = 1)
sum_nan_rows = sum(nan_rows)
nan_col = len(df) - df.count()
nan_col_per = (nan_col/len(df))* 100   #Present NAn % in each feature. 
#default
default_Total1 = len(df[df['default'] == 'yes'])
default_Total0 = len(df[df['default'] == 'no'])
default_total = len(df['default']) 
default_Total1_Proportional = round((default_Total1 / default_total) * 100 , 2)
del df['default']
#loan
loan_Total1 = len(df[df['loan'] == 'yes'])
loan__Total0 = len(df[df['loan'] == 'no'])
loan_total = len(df['loan']) 
loan_Total1_Proportional = round((loan_Total1 / loan_total) * 100 , 2)
df['loan'] = df['loan'].fillna('no')
#housing
house_Total1 = len(df[df['housing'] == 'yes'])
house_Total0 = len(df[df['housing'] == 'no'])
house_total = len(df['housing']) 
house_Total1_Proportional = round((house_Total1 / house_total) * 100 , 2)
df = df[pd.notnull(df['housing'])] 
#job
# =============================================================================
# Our hypothesn here is that ‘job’ is influenced 
# by the ‘education’ of a person. Hence, 
# we can infer ‘job’ based on the education of the person.
#  Moreover, since we are just filling the missing values,
#  we are not much concerned about the causal inference.
#  We, therefore, can use the job to predict education.
# 
# def cross_features(df, feature1, feature2):
#     jobs = list(df[feature1]).unique()
#     education = list(df[feature2]).unique()
#     cross_df = []
#     for i in education:
#         dfe = df[df[feature2] == education]
#         dfejob = dfe.groupby(feature1).count()[feature2]
#         cross_df.append(dfejob)
#     cross_matrix =pd.concat(cross_df, axis = 1)  
#     cross_matrix.columns = education
#     cross_matrix = cross_matrix.fillna(0)
#     return cross_matrix
# 
# cross_features(df,'job' ,'education') 
# =============================================================================
#job
df['job']=df.job.replace('entrepreneur', 'self-employed')
df.job.replace(['retired', 'unemployed'], 'no_active_income', inplace=True)
df.job.replace(['services', 'housemaid'], 'services', inplace=True)
df.job.replace(['admin.', 'management'], 'administration_management', inplace=True)
df = df[pd.notnull(df['job'])]
#education 
df['education']=df.education.replace(['basic.6y','basic.4y', 'basic.9y'], 'basic')
education_dic={'illiterate': 0,'basic' : 1,'high.school' : 2,'professional.course' : 3,'university.degree' : 4}
df['education']=df.education.replace(education_dic)
df = df[pd.notnull(df['education'])] 
#Impute by mean value for age & campaign
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
#age
df['age'] = imputer.fit_transform(df[['age']]) 
#campaign
df['campaign'] = imputer.fit_transform(df[['campaign']])
#marital
df = df[pd.notnull(df['marital'])]
#sum missing values
nan_rows_summary = df.isnull().sum(axis = 1)
nan_col_summary = len(df) - df.count()
nan_col_per_summary = (nan_col_summary/len(df))* 100  
sum_nan_rows_summary = sum(nan_rows_summary)
data_lost_per = 100- round(len(df)/len(df_copy_1),2)*100 
#%% Pdays - Need to be care
plt.hist(df.loc[df.pdays != 999, 'pdays'])
pd.crosstab(df['pdays'],df['poutcome'], values=df['age'], aggfunc='count', normalize=True)
# =============================================================================
# As we can see from the above table, the majority of the values for 'pdays'
#  are missing. The majority of these missing values occur when the 'poutcome'
#  is 'non-existent'. This means that the majority of the values in 'pdays'
#  are missing because the customer was never contacted before. To deal with 
#  this variable, we removed the numerical variable 'pdays' and replaced it
#  with categorical variables with following categories: p_days_missing, 
#  pdays_less_5, pdays_bet_5_15, and pdays_greater_15.
# =============================================================================
#Add new categorical variables to our dataframe.
df['pdays_missing'] = 0;
df['pdays_less_5'] = 0;
df['pdays_greater_15'] = 0;
df['pdays_bet_5_15'] = 0;
df['pdays_missing'][df['pdays']==999] = 1;
df['pdays_less_5'][df['pdays']<5] = 1;
df['pdays_greater_15'][(df['pdays']>15) & (df['pdays']<999)] = 1;
df['pdays_bet_5_15'][(df['pdays']>=5)&(df['pdays']<=15)]= 1;
df= df.drop(['pdays','pdays_less_5'], axis=1);
#Since we have many categorical variables, dummy variables needs to be created for those vaiables.
#%% Convert other Series from yes or no to binary
df['housing'] = df.housing.map(dict(yes=1, no=0));
df['loan'] = df.loan.map(dict(yes=1, no=0));
df=df.rename(columns = {'contact':'contact_by_cellular'})
df['contact_by_cellular'] = df.contact_by_cellular.map(dict(cellular = 1, telephone = 0))
df['poutcome'] = df.poutcome.map(dict(success = 1, nonexistent = 0,failure=-1))
print("Now the total NaN rows = " + str(sum(df.isna().sum())))
#%%correlation heat map
f, ax = plt.subplots(figsize=(11, 9))
# Separate both dataframes into 
numeric_df = df.select_dtypes(exclude="object")
# categorical_df = df.select_dtypes(include="object")
cor = numeric_df.corr().round(1)
mask = np.zeros_like(cor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.title("Correlation Matrix", fontsize=16)
heatmap=sns.heatmap(cor,mask=mask,annot=True,center=0,cmap=cmap,square=True, linewidths=.5, cbar_kws={"shrink": .5})
# =============================================================================
# we can see from heatmap that the highest correlate features (abs(0.8) and above)
heatmap=sns.heatmap(abs(cor),mask=mask,annot=True,center=0,cmap=cmap,square=True, linewidths=.5, cbar_kws={"shrink": .5})
# the features the economic features: ["nr.employes"-"emp.var.rate"],["cons.price.idx"-"emp.var.rate"]
# ["euribor3m"-"emp.var.rate"],["nr.employes"-"euribor3m"]
# we delete the features with the most cross feature values
# with highest correlation score:
# =============================================================================
df=df.drop(["nr.employed","emp.var.rate"],axis=1)
#%%#BOX PLOT
#age
sns.boxplot(y="age",data=numeric_df)
 
#%%
plt.close('all')
from sklearn.preprocessing import MinMaxScaler   
scaler = MinMaxScaler((-1,1))
normalized_df_data =scaler.fit_transform(numeric_df.values)
X=numeric_df
# =============================================================================
#%%The optimal value for epsilon will be found at the point of maximum curvature.
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
#%%eps = the elbow of neigh
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=1.5, min_samples=5).fit(X)
labels=db.labels_
clusterNum=len(set(labels))
#%%
df["cluster_Db"]=labels
realClusterNum=len(set(labels))-(1 if -1 in labels else 0)
noise=df.cluster_Db.value_counts()[-1]
noise_percentage=round(100*noise/df.shape[0],0)
