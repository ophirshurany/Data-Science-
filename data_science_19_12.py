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
#section  1
df = pd.read_csv("bank.csv",sep='|',encoding='utf8');
df=df.drop('Unnamed: 0',axis=1)
#view first 5 rows in df 
df.head()
#%%
#section 2.1, 2.2 , 2.3
#presenting all columns, number of rows and type
df.info()
#feature statistics for numerical categories
df.describe()
#%%
#section 2.4.1
df['y'] = df.y.map(dict(yes=1, no=0))
#section 2.4.2 
#Convert the month list to 4 binary quarters column 
months=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'];
Q = [1,1,1,1,2,2,2,3,3,3,4,4,4];month_dic=dict(zip(months,Q))
df['month']=df.month.replace(month_dic)
df=pd.get_dummies(df, columns=['month'],prefix='Q')
#%% 
#2.4.3
#Convert values from yes or no to binary
df['default'] = df.default.map(dict(yes=1, no=0))
df['housing'] = df.housing.map(dict(yes=1, no=0))
df['loan'] = df.loan.map(dict(yes=1, no=0))
#ceating age bins
df['age_by_decade'] = pd.cut(x=df['age'], 
                      bins=[10 , 19 , 29, 39 , 49 , 59 ,
                            69 , 79 , 89 , 99 , 109], 
                      labels=['10s' , '20s', '30s', '40s' , '50s'
                             , '60s' , '70s', '80s' , '90s' , '100s'])
cols = df.columns.tolist()
cols_new_order = cols[24:25] + cols[1:19] + cols [20:23] + cols[19:20] #reordering the columns +delete Q4 col in order to avoid dummy variable trap
df = df[cols_new_order] 
#%%education level rank ordering
df['education']=df.education.replace(['basic.6y','basic.4y', 'basic.9y'], 'basic')
df['education']=df.education.replace([0], 'illiterate')
df['education']=df.education.replace([1], 'basic')
df['education']=df.education.replace([2], 'high.school')
df['education']=df.education.replace([3], 'professional.course')
df['education']=df.education.replace([4], 'university.degree')                                                     
set(df['education'])
#create dummy variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_df = LabelEncoder()
df[:, 'martial'] = labelencoder_df.fit_transform(df[:,['martial'] )

"""shurani 20.12"""                                                    
#%% Pdays manipulations
df['pdays_missing'] = 0
df['pdays_less_5'] = 0
df['pdays_greater_15'] = 0
df['pdays_bet_5_15'] = 0
df['pdays_missing'][df['pdays']==999] = 1
df['pdays_less_5'][df['pdays']<5] = 1
df['pdays_greater_15'][(df['pdays']>15) & (df['pdays']<999)] = 1
df['pdays_bet_5_15'][(df['pdays']>=5)&(df['pdays']<=15)]= 1
df_dropped_pdays = df.drop('pdays', axis=1);
#%%Convert Categorial to numeric
# df['day_of_week']=df.day_of_week.astype('category').cat.codes
# df['contact']=df.contact.astype('category').cat.codes
# df['poutcome']=df.poutcome.astype('category').cat.codes
# #Convert Categorial to numeric and remains NaN
# df['education'] = df.education.astype('category').cat.codes
# df.education.replace({-1: np.nan}, inplace=True)
# df['marital']=df.marital.astype('category').cat.codes
# df.marital.replace({-1: np.nan}, inplace=True)
# df.job.replace({"unknown": np.nan}, inplace=True)
# df['job']=df.job.astype('category').cat.codes
# df.job.replace({-1: np.nan}, inplace=True)
#The significant Variables are 'education', 'job', 'housing', and 'loan'.
#sns.countplot(x='education',hue='y',data=df)
df=pd.get_dummies(df, columns=['job','day_of_week','education','contact','poutcome','marital'])
#%%correlation heat map
plt.figure(figsize=(16,16))
# Separate both dataframes into 
numeric_df = df.select_dtypes(exclude="object")
# categorical_df = df.select_dtypes(include="object")
cor = numeric_df.corr()
plt.title("Correlation Matrix", fontsize=16)
sns.heatmap(cor, annot=False,cbar=True,cmap='coolwarm')
#%%Missing Values
#show null 
print(df.isna().sum())
#BOX PLOT
lst=df.age
#lst=list(df.columns)
for col in lst:
    plt.figure()
    sns.boxplot(x='y', y=col, data=df)
#%%
from sklearn.preprocessing import MinMaxScaler, StandardScaler    
idx_numeric=[0,10,11,12,14,15,16,17,18]
scaler = MinMaxScaler()
df[df.columns[idx_numeric]] = scaler.fit_transform(df[df.columns[idx_numeric]])
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
#%%
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)
#X is np.array
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
clustering = DBSCAN(eps=3, min_samples=2).fit(df)
#%% Histograms for categorial features
cat_features = list(df.select_dtypes(include="object").columns)
for feature in cat_features:
    df_group=df.groupby("y")
    df_group_yes=df_group.get_group("yes")
    df_group_no=df_group.get_group("no")
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
#add data to skew dataset  ====================================================
sns.countplot(x='y',data=df) #before
sns.countplot(x='y',data=df)
d1=df.copy()
d2=d1[d1.y=='yes']
d3=d1[d1.y=='no']
while len(d3.y)>=len(d2.y):
    d1=pd.concat([d1, d2])
    d2=d1[d1.y=='yes']
df=d1
sns.countplot(x='y',data=df)  #after

#%% yes no distribution among features
for col in list(df.columns):
    plt.figure()
    sns.countplot(x=col,hue='y',data=df)
