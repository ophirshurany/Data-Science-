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
old_df = pd.read_csv("bank.csv",sep='|',encoding='utf8');
old_df=old_df.drop('Unnamed: 0',axis=1)
df = pd.read_csv("bank.csv",sep='|',encoding='utf8');
df=df.drop('Unnamed: 0',axis=1) #ask shir
#Dropping the duplicates
df = df.drop_duplicates()
#view first 5 rows in df 
df.head()
#%%
#section 2.1, 2.2 , 2.3
#presenting all columns, number of rows and type
df.info()
#feature statistics for numerical categories
df.describe()
#%%
#section 3 - missing data
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
""" Our hypothesn here is that ‘job’ is influenced 
by the ‘education’ of a person. Hence, 
we can infer ‘job’ based on the education of the person.
 Moreover, since we are just filling the missing values,
 we are not much concerned about the causal inference.
 We, therefore, can use the job to predict education."""

"""def cross_features(df, feature1, feature2):
    jobs = list(df[feature1]).unique()
    education = list(df[feature2]).unique()
    cross_df = []
    for i in education:
        dfe = df[df[feature2] == education]
        dfejob = dfe.groupby(feature1).count()[feature2]
        cross_df.append(dfejob)
    cross_matrix =pd.concat(cross_df, axis = 1)  
    cross_matrix.columns = education
    cross_matrix = cross_matrix.fillna(0)
    return cross_matrix





cross_features(df,'job' ,'education') """
 
#job
df.loc[(df['job'].isnull()==True) & (df['education']=='basic.4y'), 'job'] = 'blue-collar'
df.loc[(df['job'].isnull()==True) & (df['education']=='basic.6y'), 'job'] = 'blue-collar'
df.loc[(df['job'].isnull()==True) & (df['education']=='basic.9y'), 'job'] = 'blue-collar' 
df.loc[(df['age'] > 30 ) & (df['job'] == 'admin.' ) , 'job'] = 'retired'
df = df[pd.notnull(df['job'])]
#education 
df.loc[(df['education'].isnull()==True) & (df['job']=='management'), 'education'] = 'university.degree'
df.loc[(df['education'].isnull()==True) & (df['job']=='services'), 'education'] = 'high.school'
df.loc[(df['education'].isnull()==True) & (df['job']=='housemaid'), 'education'] = 'basic.4y'





#age
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
a = imputer.fit(df[['age']])
df['age'] = a .transform (df[['age']]) 

#campaign
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
a = imputer.fit(df[['campaign']])
df['campaign'] = a .transform (df[['campaign']]) 

#marital
df = df[pd.notnull(df['marital'])]


""" to check again """

df = df[pd.notnull(df['education'])] 


nan_rows_summary = df.isnull().sum(axis = 1)
nan_col_summary = len(df) - df.count()
nan_col_per_summary = (nan_col_summary/len(df))* 100  
sum_nan_rows_summary = sum(nan_rows_summary)
data_lost_per = 100- round(len(df)/len(old_df),2)*100 


#%%
#section 2.4.1
df['y'] = df.y.map(dict(yes=1, no=0))
#section 2.4.2 
#Convert the month list to 4 binary quarters column 
months=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'];
Q = [1,1,1,2,2,2,3,3,3,4,4,4];
month_dic=dict(zip(months,Q))
df['month']=df.month.replace(month_dic)
df=pd.get_dummies(df, columns=['month'],prefix='Q')
df_copy=df.copy()
#%% 
#2.4.3
#Convert values from yes or no to binary
df['housing'] = df.housing.map(dict(yes=1, no=0))
df['loan'] = df.loan.map(dict(yes=1, no=0))
#rename contact values to contact_by_cellular
df=df.rename(columns = {'contact':'contact_by_cellular'})
# convert contact_by_cellular to binary 
df['contact_by_cellular'] = df.contact_by_cellular.map(dict(cellular = 1, telephone = 0))
"""ceating age bins
df['age_by_decade'] = pd.cut(x=df['age'], 
                      bins=[10 , 19 , 29, 39 , 49 , 59 ,
                            69 , 79 , 89 , 99 , 109], 
                      labels=['10s' , '20s', '30s', '40s' , '50s'
                           , '60s' , '70s', '80s' , '90s' , '100s']) """
#education
#Combining basic school degrees
df['education']=df.education.replace(['basic.6y','basic.4y', 'basic.9y'], 'basic_school')
#education level rank ordering
df['education']=df.education.replace('illiterate', 0 )
df['education']=df.education.replace( 'basic_school' , 1)
df['education']=df.education.replace('high.school' , 2)
df['education']=df.education.replace('professional.course' , 3) 
df['education']=df.education.replace( 'university.degree' , 4)                                                     
set(df['education'])
#Combining entrepreneurs and self-employed into self-employed
df.job.replace(['entrepreneur', 'self-employed'], 'self-employed', inplace=True)
#Combining administrative and management jobs into admin_management
df.job.replace(['admin.', 'management'], 'administration_management', inplace=True)
#Combining retired and unemployed into no_active_income
df.job.replace(['retired', 'unemployed'], 'no_active_income', inplace=True)
#Combining services and housemaid into services
df.job.replace(['services', 'housemaid'], 'services', inplace=True)
#Combining single and divorced into single
df.marital.replace(['single', 'divorced'], 'single', inplace=True)

#convert categorical variables to dummy
df = pd.get_dummies(df , columns = ['job', 'marital' ,'day_of_week', 'poutcome'] , drop_first = True )


#%%2.4.1 - coreeltion between variables
""" The variable “duration” will need to be dropped before we start building a predictive model
 because it highly affects the output target (e.g., if duration=0 then y=”no”). 
 Yet, the duration is not known before a call is performed. """
del df['duration']
#Delete Q4 in order to avoid dummy variable trap
del df['Q_4']
cols = df.columns.tolist()
#making y the last columns
cols_new_order = cols[0:13] + cols[14:25] + cols[26:28] +cols[25:26] + cols [13:14]
df = df[cols_new_order] 
#Check Unique values of all the column
for i in df.columns:
  print(i)
  print(df[i].unique())
  print('---'*20)

# saturday with shurany  
  
  
  

#Calculate correlations between numeric features
plt.figure(figsize=(16,16))
numeric_df = df.select_dtypes(exclude="object")
numeric_df = numeric_df.select_dtypes(exclude="uint8")
num_cor_matrix = numeric_df.corr()
plt.title("Correlation numeric vars Matrix", fontsize=16)
sns.heatmap(num_cor_matrix, annot=False,cbar=True,cmap='coolwarm')
"""nr.employed and emp.var.rate are  highly  corelated. 
there is stronger corlation between the y and nr.employed than y and emp.var.rate
 because that we will remove emp.var.rate """
del df['emp.var.rate']
"""nr.employed and euribor3m are highly  corelated. 
there is stronger corlation between the y and nr.employed than y and euribor3m 
#because that we will remove euribor3m """
del df['euribor3m']
#Calculate  new correlations between numeric features
plt.figure(figsize=(16,16))
numeric_df = df.select_dtypes(exclude="object")
numeric_df = numeric_df.select_dtypes(exclude="uint8")
num_cor_new_matrix = numeric_df.corr()
plt.title("New Correlation numeric vars Matrix", fontsize=16)
sns.heatmap(num_cor_new_matrix, annot=False,cbar=True,cmap='coolwarm')
#Calculate   correlations between categorical features
plt.figure(figsize=(16,16))
categorical_df = df.select_dtypes(exclude="int64")
categorical_df = categorical_df.select_dtypes(exclude="float64")
categorical_cor_matrix = categorical_df.corr()
plt.title("Correlation categorical vars Matrix", fontsize=16)
sns.heatmap(categorical_cor_matrix, annot=False,cbar=True,cmap='coolwarm')
#normalizng the data for the db san and pca
from sklearn import preprocessing   
#separate the data from the target attributes
#nor normalize the data for the pca -4.2
normalized_df=(df-df.mean())/df.std()
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)



"""shurani box plot and db scan"""
#%%#BOX PLOT
for col in numeric_df.columns:
    plt.figure()
    sns.boxplot(x=numeric_df[col])
#%%
from sklearn.preprocessing import MinMaxScaler   
scaler = MinMaxScaler((-1,1))
normalized_df_data =scaler.fit_transform(numeric_df.values)
X=numeric_df
X=X.dropna()
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


