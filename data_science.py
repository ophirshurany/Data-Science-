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
#%%1. create dataframe
df = pd.read_csv("bank.csv",sep='|',encoding='utf8');
#drop duplicate data
df = df.drop_duplicates('Unnamed: 0',keep=False)
#drop #rows
df=df.drop('Unnamed: 0',axis=1)
df_copy_original=df #Keep original

#%%2.1. view first 5 rows in df
df.head()
#2.2. presenting all columns, number of rows and type
df.info()
#2.3. feature statistics for numerical categories
df.describe()
#%% Histograms for categorial features
categorcial_variables =list(df.select_dtypes(include="object").columns)
for feature in categorcial_variables:
    plt.figure(figsize=(14, 9))
    sns.countplot(x=feature,data=df)
    plt.title("Feature Histogram - " + feature,fontsize='xx-large', fontweight='bold')
    plt.ylabel("Count")
    plt.xlabel("Feature unique values")
    plt.tight_layout()
#%% Histograms for Numeric features
num_features = list(df.select_dtypes(exclude="object").columns)
for feature in num_features:
#devide for economic 
    plt.figure(figsize=(14, 9))
    sns.distplot(df[feature].dropna(),kde=False)
    plt.title("Feature Histogram - " + feature,fontsize='xx-large', fontweight='bold')
    plt.ylabel("Count")
    plt.xlabel(feature)
    plt.tight_layout()
#%%2.4
#2.4.1.change "yes" or "no" to 1 or 0
df['y'] = df.y.map(dict(yes=1, no=0))
#2.4.2. Convert the month list to 4 binary quarters column 
months=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'];
Q = [1,1,1,1,2,2,2,3,3,3,4,4,4];month_dic=dict(zip(months,Q))
df['month']=df.month.replace(month_dic)
df=pd.get_dummies(df, columns=['month'],prefix='Q')
#2.4.3. convert categorial features to numeric and drop the number of variables
#education
df['education']=df.education.replace(['basic.6y','basic.4y', 'basic.9y'], 'basic')
#for education it makes sense to use ranking
education_dic={'illiterate': 0,'basic' : 1,'high.school' : 2,'professional.course' : 3,'university.degree' : 4}
df['education']=df.education.replace(education_dic)
#job
df.job.replace(['admin.', 'management'], 'administration_management', inplace=True)
df.loc[(df['age'] > 60 ) & (df['job'] == 'admin.' ) , 'job'] = 'retired'
df.job.replace(['retired', 'unemployed'], 'no_active_income', inplace=True)
df.job.replace('housemaid', 'services')
df['job']=df.job.replace('entrepreneur', 'self-employed')
# Convert other Series from yes or no to binary
df['housing'] = df.housing.map(dict(yes=1, no=0));
df['loan'] = df.loan.map(dict(yes=1, no=0));
df=df.rename(columns = {'contact':'contact_by_cellular'})
df['contact_by_cellular'] = df.contact_by_cellular.map(dict(cellular = 1, telephone = 0))
#%%3. - missing data
print("Total NaN rows = " + str(sum(df.isna().sum())))
 #Present NAn % in each feature. 
(100*df.isna().sum()/df.shape[0]).round(1)
#we need to see how values are distributed:
#first, we convert unknown values from NaN so they will be countable as unknown:
df['default'] = df.default.replace(np.nan,'unknown',regex=True)
#default
pd.crosstab(df['y'],df['default'],dropna=True).apply(lambda r: r/r.sum(), axis=1).round(4)
#most of No are at default, so we cant really learn from it. then, it will be deleted
df=df.drop("default",axis=1)
#loan
df['loan'] = df.loan.replace(np.nan,'unknown',regex=True)
pd.crosstab(df['y'],df['loan']).apply(lambda r: r/r.sum(), axis=1).round(2)
df['loan'] = df['loan'].replace('unknown',0)
#housing
df['housing'] = df.housing.replace(np.nan,'unknown',regex=True)
pd.crosstab(df['y'],df['housing']).apply(lambda r: r/r.sum(), axis=1).round(2)
#values distribute practicly evenly, therefore we can delete uknowns:
df = df[df.housing != "unknown"]
# =============================================================================
# Our hypothesn here is that ‘job’ is influenced 
# by the ‘education’ of a person. Hence, 
# we can infer ‘job’ based on the education of the person.
#  Moreover, since we are just filling the missing values,
#  we are not much concerned about the causal inference.
#  We, therefore, can use the job to predict education.
# =============================================================================
#to infer the missing values in 'job' and 'education', we make use of the cross-tabulation between 'job' and 'education'.
a=pd.crosstab(df['job'], df['education'], rownames=['job'], colnames=['education'])
# =============================================================================
# While imputing the values for job and education, we were cognizant of the fact that
# the correlations should make real world sense. If it didn't 
# make real world sense, we didn't replace the missing values.
# =============================================================================
#job
df['job'] = df.job.replace(np.nan,'unknown',regex=True)
df.loc[(df['job']=='unknown') & (df['education']==1), 'job'] = 'blue-collar'
df.loc[(df['job']=='unknown') & (df['education']==3), 'job'] = 'technician'
#education 
df['education'] = df.education.replace(np.nan,'unknown',regex=True)
df.loc[(df['education']=='unknown') & (df['job']=='management'), 'education'] = 4
df.loc[(df['education']=='unknown') & (df['job']=='services'), 'education'] = 2
#Impute by mean value for age & campaign
#df = df.replace('unknown',np.nan)
#age
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
df["age"] = imputer.fit_transform(df[['age']])
#campaign
df['campaign'] = imputer.fit_transform(df[['campaign']])
#marital
# Examine the missing values in 'pdays'
plt.figure()
plt.hist(df.loc[df.pdays != 999, 'pdays'])
plt.title("Pdays Data Distribution Without 999", fontsize='xx-large', fontweight='bold')
plt.xlabel("Pdays")
plt.ylabel("Count")
# =============================================================================
# Filtered out missing values encoded with an out-of-range value when
#  plotting the histogram of values in order to properly understand
#  the distribution of the known values. Here, histograms were
#  created using matplotlib.
# =============================================================================
pd.crosstab(df['pdays'],df['poutcome'], values=df['age'], aggfunc='count')
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
#convert categorical variables to dummy
df = pd.get_dummies(df , columns = ['job', 'marital' ,'day_of_week', 'poutcome'])
print("Now the total NaN rows = " + str(sum((df == 'unknown').sum())))
df = df[df != "unknown"]
print("we now remove all other NaN")
df = df.dropna()
print("Number of deleted rows = " + str(df_copy_original.shape[0]-df.shape[0]))
print("only "+ str(round(100*(df_copy_original.shape[0]-df.shape[0])/df.shape[0],1))+" %")
print("Finally, the total NaN rows = " + str(sum(df.isna().sum())))
#%%2.5. correlation heat map
cor = df.corr().round(1)
mask = np.zeros_like(cor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.figure()
heatmap=sns.heatmap(cor,mask=mask,annot=True,annot_kws={"size": 7},
                    center=0,cmap=cmap,square=True, linewidths=.5,
                    cbar_kws={"shrink": .5},yticklabels=1,xticklabels=1)
plt.title("Correlation Matrix", fontsize='xx-large', fontweight='bold')
# =============================================================================
# we can see from heatmap that the highest correlate features (abs(0.8) and above)
# the features the economic features: ["nr.employes"-"emp.var.rate"],["cons.price.idx"-"emp.var.rate"]
# ["euribor3m"-"emp.var.rate"],["nr.employes"-"euribor3m"]
# nr.employed and emp.var.rate are  highly  corelated and also nr.employed
# and euribor3m are highly  corelated.  
#  because that we will remove emp.var.rate and euribor3m
# =============================================================================
#duration =====================================================================
# The variable “duration” will need to be dropped before we start building a predictive model
#  because it highly affects the output target (e.g., if duration=0 then y=”no”). 
#  Yet, the duration is not known before a call is performed.
# =============================================================================
#Q4
#Delete Q4 in order to avoid dummy variable trap
df=df.drop(["duration","Q_4","euribor3m","emp.var.rate","poutcome_nonexistent","marital_single"],axis=1)
df_copy_feature_filtered=df_copy_original.drop(["duration","euribor3m","emp.var.rate"],axis=1)
#now we want to see the updated correlation matrix
cor = df.corr().round(1)
mask = np.zeros_like(cor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.figure()
heatmap=sns.heatmap(cor,mask=mask,annot=True,annot_kws={"size": 7},
                    center=0,cmap=cmap,square=True, linewidths=.5,
                    cbar_kws={"shrink": .5},yticklabels=1,xticklabels=1)
plt.title("Updated Correlation Matrix", fontsize='xx-large', fontweight='bold')
#%%#BOX PLOT
#Outliers: Outliers are defined as 1.5 x Q3 value (75th percentile).
feature_lst=["cons.price.idx","nr.employed","cons.conf.idx","age","campaign","previous"]
df_with_outliers=df.copy()
#cons.price.idx
sns.boxplot(x='y', y="cons.price.idx", data=df_with_outliers)
#There are no outliers for this feature.
#nr.employed
sns.boxplot(x='y', y="nr.employed", data=df_with_outliers)
#There are no outliers for this feature.
#cons.conf.idx
sns.boxplot(x='y', y="cons.conf.idx", data=df_with_outliers)
#There are some unusual results in the target variable "no", 
#but these do not significantly exceed the upper limit. Then, 
#they fit the upper bound of the target variable "yes". 
#Therefore, we chose to leave it.
sns.boxplot(x='y', y="age", data=df_with_outliers)
sns.boxplot(x='y', y="campaign", data=df_with_outliers)
# =============================================================================
# We have outliers as max('age') and max('campaign') > 1.5Q3('age')
# and >1.5Q3('campaign') respectively.
# But we also see that the value of these outliers are not so unrealistic
# (max('age')=98 and max('campaign')=56).
# Hence, we need not remove them since the prediction model 
# should represent the real world. This improves the 
# generalizability of the model and makes it robust 
# for real world situations. 
# The outliers, therefore, are not removed.
# =============================================================================
sns.boxplot(x='y', y="previous", data=df_with_outliers)
# =============================================================================
# This variable has many unusual results, from the database, The unusual 
# results belong to many calls made to a customer and therefore the outlier
#  results are much higher. We decided to sift the top results that 
#  exceed 3 times the upper limit, leaving the other 
#  results less than the top limit.
df_outliers=df_with_outliers[["previous"]]
# =============================================================================
#Normalize features
plt.close('all')
from sklearn.preprocessing import MinMaxScaler   
numeric_df = df.select_dtypes(exclude="object")
scaler = MinMaxScaler()
normalized_df_data =scaler.fit_transform(numeric_df.values)
x_scaled=pd.DataFrame(normalized_df_data,columns=numeric_df.columns)
# =============================================================================
#%%5.1
#The optimal value for epsilon will be found at the point of maximum curvature.
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(x_scaled)
distances, indices = nbrs.kneighbors(x_scaled)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.title("Find  the  optimal "+r'$  \varepsilon$',fontsize='xx-large', fontweight='bold')
plt.ylabel("epsilon")
plt.xlabel("Feature unique values")
plt.plot([37210], [1.1], 'ro')
plt.annotate('Optimal '+r'$\varepsilon$', (37210,1.1),
            xytext=(0.8, 0.9), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=16,
            horizontalalignment='right', verticalalignment='top')
plt.tight_layout()
#eps = the best epsilon is at the "elbow" of NearestNeighbors graph
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=1.1,min_samples=5).fit(x_scaled)
labels=db.labels_
clusterNum=len(set(labels))
noise=np.count_nonzero(labels == -1)
noise_percentage=round(100*noise/df.shape[0],0)
#%%
df["cluster_Db"]=labels
realClusterNum=len(set(labels))-(1 if -1 in labels else 0)
df = df[df.cluster_Db != -1]
Num_outliers_1st=df_with_outliers.shape[0]-df.shape[0]
noise_percentage=round(100*Num_outliers_1st/df.shape[0],0)
#5.2 Lots of clusters means low number of noise, therefore low number of outliers.
#5.3 - Another method to remove outliers
from scipy import stats
z = np.abs(stats.zscore(df_outliers))
#define a threshold to identify an outlier
threshold = 3
df_outliers=df_outliers[(z < threshold).all(axis=1)]
Num_outliers_2nd=df_with_outliers.shape[0]-df_outliers.shape[0]
