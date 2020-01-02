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
df =old_df.copy()
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
nan_rows = df.isnull().sum(axis = 1)
sum_nan_rows = sum(nan_rows)            #Present total rows with NA. 
print (sum_nan_rows)
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
df = df[pd.notnull(df['education'])] 
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
#pdays

# """two variables define if a respondent is a new or existing customer:
#  — poutcomeand(For the new customer the value would be - “nonexistent” )"and pday (the value would be 999). 
#  Let’s see if we have the same number of respondents for each of these levels:"""
# print (df.poutcome.value_counts()['nonexistent'] , df.pdays.value_counts()[999]  ) 
# """ #The numbers vary significantly when logically they should be the same.
# Let’s check another variable- ‘previous’, which contains the number of customer contacts performed before
#  the current campaign. A zero value would indicate the new customers """
# print (df.previous.value_counts()[0] ) 
# """The number of the new customers in “previous” matches the number of the “nonexistent”
#  respondents in the “poutcome” variable exactly. the difference between them is 0: """ 
# print (df.previous.value_counts()[0]   - df.poutcome.value_counts()['nonexistent'])
# """"Now, let’s check if the “pdays” variable has the “999” value
#  for any levels of the “poutcome” variable other than “nonexistent” """
# #Filtering the 'poutcome' and 'pdays' variables
# not_matching = df.loc[( (df['pdays'] == 999) & (df['poutcome'] != 'nonexistent') )]
# #Counting the values
# print(not_matching.poutcome.value_counts()['failure'])
# "This is exactly the difference in the counts of the values for “poutcome” and “pdays” variables"
# print(df.pdays.value_counts()[999] -df.poutcome.value_counts()['nonexistent'] )
# """it looks like 3,812 entries of the variable “pdays” are mistakenly labeled as “999
# "we will consider the “pdays” values of “999” for the rows that have the “poutcome
#  of “failure” to be missing variables. """
#  #Getting the positions of the mistakenly labeled 'pdays'
# ind_999 = df.loc[(df['pdays'] == 999) & (df['poutcome'] != 'nonexistent')]['pdays'].index.values
# amount_ind_999 = len(ind_999)
# wrong_999_proportion = round(amount_ind_999  /len(df),2) *100
# """we decied to remove the "wrong 999" beacause we cant figure what is the right valuse
#  and its just 10% of the data. keeping this wrong values wold make a wrong predtiction for the y values
# ( wec checked it)  """
# #Assigning NaNs instead of '999'
# df.loc[ind_999, 'pdays'] = np.nan
# df = df[pd.notnull(df['pdays'])]
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
#%% 
#2.4.3
#Convert values from yes or no to binary
df['housing'] = df.housing.map(dict(yes=1, no=0))
df['loan'] = df.loan.map(dict(yes=1, no=0))
#rename contact values to contact_by_cellular
df=df.rename(columns = {'contact':'contact_by_cellular'})
# convert contact_by_cellular to binary 
df['contact_by_cellular'] = df.contact_by_cellular.map(dict(cellular = 1, telephone = 0))

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
cols_new_order = cols[0:12]  + cols[13:28] + cols[29:31] + cols[28:29]  + cols[31:33] + cols[12:13] #fix later 
df = df[cols_new_order] 
#Check Unique values of all the column
for i in df.columns:
  print(i)
  print(df[i].unique())
  print('---'*20)
#Calculate correlations between numeric features
plt.figure(figsize=(16,16))
numeric_df = df.select_dtypes(exclude="object")
numeric_df = numeric_df.select_dtypes(exclude="uint8")
num_cor_matrix = numeric_df.corr().round(1)
plt.title("Correlation numeric vars Matrix", fontsize=16)
mask = np.zeros_like(num_cor_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.title("Correlation Matrix ", fontsize=16)
heatmap=sns.heatmap(num_cor_matrix,mask=mask,annot=True,center=0,cmap=cmap,square=True, linewidths=.5, cbar_kws={"shrink": .5})
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
"""We chhosed to used Standardization and not  Min-Max scaling.
In clustering analyses, standardization may be especially crucial in order to 
similarities between features based on certain distance measures.
Another prominent example is the PCA, where we usually prefer standardization over Min-Max scaling, 
since we are interested in the components that maximize the variance""" 
df_numeric = df.iloc[:,0:10]
numeric_cols = df_numeric.columns.tolist()
df_categorial = df.iloc[:,10:]
from sklearn import preprocessing   
min_max_scaler = preprocessing.MinMaxScaler()
df_numeric_scaled = min_max_scaler.fit_transform(df_numeric)
df_numeric_scaled= pd.DataFrame(df_numeric_scaled)
df_numeric_scaled.columns = numeric_cols 
df_numeric_scaled.reset_index(inplace=True, drop=True) 
df_categorial.reset_index(inplace=True, drop=True) 
df = pd.concat([df_numeric_scaled, df_categorial] , axis = 1  )

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
df_outliers = df_with_outliers[["previous"]]
#%%5.1
#The optimal value for epsilon will be found at the point of maximum curvature.
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(round(df,1))
distances, indices = nbrs.kneighbors(df)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.plot(32208,2.4,'ro')
#eps = the elbow of neigh
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.8, min_samples=5).fit(df)
labels=db.labels_
clusterNum=len(set(labels))
df["cluster_Db"]=labels
realClusterNum=len(set(labels))-(1 if -1 in labels else 0)
noise=df.cluster_Db.value_counts()[-1]
noise_percentage=round(100*noise/df.shape[0],0)
df_DBSCAN = df[df.cluster_Db != -1]
Num_outliers_1st=df_with_outliers.shape[0]-df_DBSCAN.shape[0]
del df_DBSCAN["cluster_Db"]

#%%
#5.2 Lots of clusters means low number of noise, therefore low number of outliers.
#%%
#5.3 - Another method to remove outliers
from scipy import stats
z = np.abs(stats.zscore(df_outliers))
#define a threshold to identify an outlier
threshold = 4
df_outliers=df_outliers[(z < threshold).all(axis=1)]
Num_outliers_2nd=df_with_outliers.shape[0]-df_outliers.shape[0]
df_w=df.drop("cluster_Db",axis=1)
del df['cluster_Db']
#%%
# Logistic Regression to the Training set include outliers
X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values
onesYCount = np.count_nonzero(df['y'])
prop_ones_y =  onesYCount/ len(df['y'])
# we can see that the data is unproportional
print (prop_ones_y)  
X_without_outliers = X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values
#%%
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Splitting the dataset without_outliers after dbs into the Training set and Test set
X_DBS = df_DBSCAN.iloc[:, 0:-1].values
y_DBS = df_DBSCAN.iloc[:, -1].values
X_DBS_train, X_DBS_test, y_DBS_train, y_DBS_test = train_test_split(X_DBS, y_DBS, test_size = 0.2, random_state = 0)
# Splitting the dataset without outliers after droping threshold the Training set and Test set
X_w = df_w.iloc[:, 0:-1].values
y_w = df_w.iloc[:, -1].values
X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(X_w, y_w, test_size = 0.2, random_state = 0)
#%%
# Fitting Logistic Regression to the Training set include outliers
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
LR_y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
LR_cm= confusion_matrix(y_test, LR_y_pred)
col_names = ['a_0','a_1' ]
index_names = ['p_0','p_1' ]
LR_accuracy = (LR_cm[0][0] + LR_cm[1][1]) /np.sum(LR_cm)
print(LR_accuracy )
LR_precision =  (LR_cm[1][1] /np.sum(LR_cm[: , 1])) #TP/(TP+FN) Predicted 1 and actual 1 /(total actual 1)
print (LR_precision) 
LR_recall = LR_cm[1][1] /np.sum((LR_cm[1 , :])) #TP/(TP+FN)  Predicted 1 and actual 1   / total Predicted 1
print (LR_recall) 
#Creating the LR classification report
from sklearn.metrics import classification_report , roc_auc_score , roc_curve
print ("Logistic Regression classification report")
print(classification_report(y_test, LR_y_pred)) 
#Obtaining the ROC score
LR_roc_auc = roc_auc_score(y_test, LR_y_pred)
#Obtaining false and true positives & thresholds
fpr, tpr, thresholds = roc_curve(y_test, LR_y_pred)
#Plotting the curve
plt.plot(fpr, tpr, label=' Logistic Regression (AUC = %0.03f)' % LR_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
" axis x =TP RATE = FP/ (FP +TN) = by mistake predict as 1/ total 0. low x is good"
" axis y =FP RATE = TP/(TP + FN) = said it 1 and was right /total 1. high y is good."
plt.title('ROC curve for Logistic regression')
plt.legend(loc="upper left")
plt.show()
#%%
# Fitting Logistic Regression to the Training set after DBS
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_DBS_train, y_DBS_train)
# Predicting the Test set results
LR_y_DBS_pred= classifier.predict(X_DBS_test)
# Making the Confusion Matrix
LR_DBS_cm= confusion_matrix(y_DBS_test, LR_y_DBS_pred)
col_names = ['a_0','a_1' ]
index_names = ['p_0','p_1' ]
LR_DBS_accuracy = (LR_DBS_cm[0][0] + LR_DBS_cm[1][1]) /np.sum(LR_DBS_cm)
print(LR_DBS_accuracy )
LR_DBS_precision =  (LR_DBS_cm[1][1] /np.sum(LR_DBS_cm[: , 1])) #TP/(TP+FN) Predicted 1 and actual 1 /(total actual 1)
print (LR_DBS_precision) 
LR_DBS_recall = LR_cm[1][1] /np.sum((LR_cm[1 , :])) #TP/(TP+FN)  Predicted 1 and actual 1   / total Predicted 1
print (LR_DBS_recall) 
#Creating the LR classification report
print ("Logistic Regression without outliers classification report")
print(classification_report(y_DBS_test, LR_y_DBS_pred)) 
#Obtaining the ROC score
LR_DBS_roc_auc = roc_auc_score(y_DBS_test, LR_y_DBS_pred)
#Obtaining false and true positives & thresholds
fpr, tpr, thresholds = roc_curve(y_DBS_test, LR_y_DBS_pred)
#Plotting the curve
plt.plot(fpr, tpr, label=' Logistic Regression without outliers(DBS) (AUC = %0.03f)' % LR_DBS_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
" axis x =TP RATE = FP/ (FP +TN) = by mistake predict as 1/ total 0. low x is good"
" axis y =FP RATE = TP/(TP + FN) = said it 1 and was right /total 1. high y is good."
plt.title('ROC curve for Logistic regression without outliers (DBS)')
plt.legend(loc="upper left")
plt.show()
#%%
# Fitting Logistic Regression to the Training set after threshold
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_w_train, y_w_train)
# Predicting the Test set results
LR_y_w_pred= classifier.predict(X_w_test)
# Making the Confusion Matrix
LR_w_cm= confusion_matrix(y_w_test, LR_y_w_pred)
col_names = ['a_0','a_1' ]
index_names = ['p_0','p_1' ]
LR_w_accuracy = (LR_w_cm[0][0] + LR_w_cm[1][1]) /np.sum(LR_w_cm)
print(LR_w_accuracy )
LR_w_precision =  (LR_w_cm[1][1] /np.sum(LR_w_cm[: , 1])) #TP/(TP+FN) Predicted 1 and actual 1 /(total actual 1)
print (LR_w_precision) 
LR_w_recall = LR_cm[1][1] /np.sum((LR_cm[1 , :])) #TP/(TP+FN)  Predicted 1 and actual 1   / total Predicted 1
print (LR_w_recall) 
#Creating the LR classification report
print ("Logistic Regression without outliers classification report")
print(classification_report(y_w_test, LR_y_w_pred)) 
#Obtaining the ROC score
LR_w_roc_auc = roc_auc_score(y_w_test, LR_y_w_pred)
#Obtaining false and true positives & thresholds
fpr, tpr, thresholds = roc_curve(y_w_test, LR_y_w_pred)
#Plotting the curve
plt.plot(fpr, tpr, label=' Logistic Regression without outliers(threshold) (AUC = %0.03f)' % LR_w_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
" axis x =TP RATE = FP/ (FP +TN) = by mistake predict as 1/ total 0. low x is good"
" axis y =FP RATE = TP/(TP + FN) = said it 1 and was right /total 1. high y is good."
plt.title('ROC curve for Logistic regression without outliers (threshold)')
plt.legend(loc="upper left")
plt.show()
# we got the best result with the orignial DB, threfore we will use him
#%%
# Fitting  Regression Tree and GradientBoostin to the Training set
#---------------------------------------------------------------------
#Setting up pipelines with a StandardScaler function to normalize the variables
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier


#Setting up pipelines with a StandardScaler function to normalize the variables
pipelines = {
    'rf' : make_pipeline(StandardScaler(), 
                         RandomForestClassifier(random_state=42, class_weight='balanced')),
    'gb' : make_pipeline(StandardScaler(), 
                         GradientBoostingClassifier(random_state=42))
}

#Setting up the "rule of thumb" hyperparameters for the Random Forest
rf_hyperparameters = {
    'randomforestclassifier__n_estimators': [100, 200],
    'randomforestclassifier__max_features': ['auto', 'sqrt', 0.33]
}

#Setting up the "rule of thumb" hyperparameters for the Gradient Boost
gb_hyperparameters = {
    'gradientboostingclassifier__n_estimators': [100, 200],
    'gradientboostingclassifier__learning_rate': [0.05, 0.1, 0.2],
    'gradientboostingclassifier__max_depth': [1, 3, 5]
}

#Creating the dictionary of hyperparameters
hyperparameters = {
    'rf' : rf_hyperparameters,
    'gb' : gb_hyperparameters
}

#Creating an empty dictionary for fitted models
fitted_alternative_models = {}

# Looping through model pipelines, tuning each with GridSearchCV and saving it to fitted_logreg_models
for name, pipeline in pipelines.items():
    #Creating cross-validation object from pipeline and hyperparameters
    alt_model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)
    
    #Fitting the model on X_train, y_train
    alt_model.fit(X_train, y_train)
    
    #Storing the model in fitted_logreg_models[name] 
    fitted_alternative_models[name] = alt_model
    
    #Printing the status of the fitting
    print(name, 'has been fitted.')

for name, model in fitted_alternative_models.items():
    print(name, model.best_score_ )    
    
RF_y_pred= fitted_alternative_models['rf'].predict(X_test)
print("Randon foresst summary: \n " ,  classification_report(y_test, RF_y_pred))
RF_roc_auc = roc_auc_score(y_test, RF_y_pred)
print("Randon foresst auc : " ,  RF_roc_auc )
  
GB_y_pred= fitted_alternative_models['gb'].predict(X_test)
print("Gradient BoostingC summary: \n " ,  classification_report(y_test, GB_y_pred))
GB_roc_auc = roc_auc_score(y_test, GB_y_pred)
print("Gradien tBoostingC foresst auc : " ,  GB_roc_auc )
#%% KNN model
from sklearn.neighbors import KNeighborsClassifier
# search for an optimal value of K for KNN
# list of integers 1 to50
# integers we want to try
k_range = range(1,10)

# list of scores from k_range
k_scores = []

# 1. we will loop through reasonable values of k
for k in k_range:
    # 2. run KNeighborsClassifier with k neighbours
    knn = KNeighborsClassifier(n_neighbors=k)
    # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    # 4. append mean of scores for k neighbors to k_scores list
    k_scores.append(scores.mean())
print(k_scores)
# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy') 
# According this plot the chosen k would be 4
choosen_k = 4
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = choosen_k, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
# Predicting the Test set results
KNN_y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
KNN_cm= confusion_matrix(y_test, KNN_y_pred)
col_names = ['a_0','a_1' ]
index_names = ['p_0','p_1' ]
KNN_accuracy = (KNN_cm[0][0] + KNN_cm[1][1]) /np.sum(KNN_cm)
print(KNN_accuracy )
KNN_precision =  (KNN_cm[1][1] /np.sum(KNN_cm[: , 1])) #TP/(TP+FN) Predicted 1 and actual 1 /(total actual 1)
print (KNN_precision) 
KNN_recall = KNN_cm[1][1] /np.sum((KNN_cm[1 , :])) #TP/(TP+FN)  Predicted 1 and actual 1   / total Predicted 1
print (KNN_recall) 
#Creating the LR classification report
from sklearn.metrics import classification_report , roc_auc_score , roc_curve
print(classification_report(y_test, KNN_y_pred)) 
#Obtaining the ROC score
roc_auc = roc_auc_score(y_test, KNN_y_pred)
#Obtaining false and true positives & thresholds
fpr, tpr, thresholds = roc_curve(y_test, KNN_y_pred)
#Plotting the curve
plt.plot(fpr, tpr, label=' KNN (AUC = %0.03f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
" axis x =TP RATE = FP/ (FP +TN) = by mistake predict as 1/ total 0. low x is good"
" axis y =FP RATE = TP/(TP + FN) = said it 1 and was right /total 1. high y is good."
plt.title('ROC curve for Logistic regression')
plt.legend(loc="upper left")
plt.show()
KNN_cm= pd.DataFrame(KNN_cm, index=col_names, columns=index_names) 
#%%    
#Upsampling the  data 
#Separating the imbalanced observations into 2 separate datasets
from sklearn.utils import resample
df_majority = df[df['y']==0]
df_minority = df[df['y']==1]
df_majority = pd.DataFrame(df_majority)
df_minority = pd.DataFrame(df_minority)
#Upsampling the minority class
df_minority_upsampled = resample(df_minority, replace=True, n_samples=33401, random_state=42)
#Concatenating two datasets
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
#New class counts
print(df_upsampled['y'].value_counts())
print(df_upsampled['y'].unique())
#%%    
#Down sampling the  data 
from sklearn.utils import resample
np.random.seed(0)
df_zero_index = df.index[df['y'] == 0].tolist()
df_majority_dowmsampled_index = np.random.choice(df_zero_index, 24000, replace=False)
df_downsampled = df.drop(df_majority_dowmsampled_index)
df_downsampled['y'].value_counts()
   
#%%"
#we will check the best method for our model, logistic regression , on the upsampled and the downsampled data
#try logistic regression om the upbalanced data
X_up = df_upsampled.iloc[:, 0:-1].values
y_up = df_upsampled.iloc[:, -1].values
X_train_up, X_test_up, y_train_up, y_test_up = train_test_split(X_up, y_up, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_up, y_train_up)
# Predicting the Test set results
LR_y_pred_up = classifier.predict(X_test_up)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
LR_cm_up= confusion_matrix(y_test_up, LR_y_pred_up)
col_names = ['a_0','a_1' ]
index_names = ['p_0','p_1' ]
LR_accuracy_up = (LR_cm_up[0][0] + LR_cm_up[1][1]) /np.sum(LR_cm_up)
print(LR_accuracy_up )
LR_precision_up =  (LR_cm_up[1][1] /np.sum(LR_cm_up[: , 1])) #TP/(TP+FN) Predicted 1 and actual 1 /(total actual 1)
print (LR_precision_up) 
LR_recall_up = LR_cm_up[1][1] /np.sum((LR_cm_up[1 , :])) #TP/(TP+FN)  Predicted 1 and actual 1   / total Predicted 1
print (LR_recall_up) 
#Creating the LR classification report
from sklearn.metrics import classification_report , roc_auc_score , roc_curve
print(classification_report(y_test_up, LR_y_pred_up)) 
#Obtaining the ROC score
roc_auc_up = roc_auc_score(y_test_up , LR_y_pred_up)
#Obtaining false and true positives & thresholds
fpr, tpr, thresholds = roc_curve(y_test_up, LR_y_pred_up)
#Plotting the curve
plt.plot(fpr, tpr, label=' Logistic Regression UpBalanced data (AUC = %0.02f)' % roc_auc_up)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
" axis x =TP RATE = FP/ (FP +TN) = by mistake predict as 1/ total 0. low x is good"
" axis y =FP RATE = TP/(TP + FN) = said it 1 and was right /total 1. high y is good."
plt.title('ROC curve for Logistic regression')
plt.legend(loc="upper left")
plt.show()
LR_cm_up= pd.DataFrame(LR_cm_up, index=col_names, columns=index_names)
#%%
#try logistic regression om the downbalanced data
X_down = df_downsampled.iloc[:, 0:-1].values
y_down = df_downsampled.iloc[:, -1].values
X_train_down, X_test_down, y_train_down, y_test_down = train_test_split(X_down, y_down, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_down, y_train_down)
# Predicting the Test set results
LR_y_pred_down = classifier.predict(X_test_down)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
LR_cm_down= confusion_matrix(y_test_down, LR_y_pred_down)
col_names = ['a_0','a_1' ]
index_names = ['p_0','p_1' ]
LR_accuracy_down = (LR_cm_down[0][0] + LR_cm_down[1][1]) /np.sum(LR_cm_down)
print(LR_accuracy_down )
LR_precision_down =  (LR_cm_down[1][1] /np.sum(LR_cm_down[: , 1])) #TP/(TP+FN) Predicted 1 and actual 1 /(total actual 1)
print (LR_precision_down) 
LR_recall_down = LR_cm_down[1][1] /np.sum((LR_cm_down[1 , :])) #TP/(TP+FN)  Predicted 1 and actual 1   / total Predicted 1
print (LR_recall_down) 
#Creating the LR classification report
from sklearn.metrics import classification_report , roc_auc_score , roc_curve
print(classification_report(y_test_down, LR_y_pred_down)) 
#Obtaining the ROC score
roc_auc_down = roc_auc_score(y_test_down , LR_y_pred_down)
#Obtaining false and true positives & thresholds
fpr, tpr, thresholds = roc_curve(y_test_down, LR_y_pred_down)
#Plotting the curve
plt.plot(fpr, tpr, label=' Logistic Regression downBalanced data (AUC = %0.03f)' % roc_auc_down)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
" axis x =TP RATE = FP/ (FP +TN) = by mistake predict as 1/ total 0. low x is good"
" axis y =FP RATE = TP/(TP + FN) = said it 1 and was right /total 1. high y is good."
plt.title('ROC curve for Logistic regression')
plt.legend(loc="uper left")
plt.show()
LR_cm_down= pd.DataFrame(LR_cm_down, index=col_names, columns=index_names)
#the best results was on the upsampled data
#%%
#trying pca on the balanced logistic regression
from sklearn.decomposition import PCA
X_train_up_pca, X_test_up_pca, y_train_up_pca, y_test_up_pca = train_test_split(X_up, y_up, test_size = 0.2, random_state = 0)
pca_up= PCA()
X_train_up_pca = pca_up.fit_transform(X_train_up_pca)
X_test_up_pca = pca_up.transform(X_test_up_pca)
explained_variance_pca_up =pca_up.explained_variance_ratio_
#we will choose  vars with variance more than 0.1
X_train_up_pca, X_test_up_pca, y_train_up_pca, y_test_up_pca = train_test_split(X_up, y_up, test_size = 0.2, random_state = 0)
pca_up = PCA(2)
X_train_up_pca = pca_up.fit_transform(X_train_up_pca)
X_test_up_pca = pca_up.transform(X_test_up_pca)
explained_variance_pca_up =pca_up.explained_variance_ratio_
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_up_pca, y_train_up_pca)
# Predicting the Test set results
LR_y_pred_up_pca = classifier.predict(X_test_up_pca)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
LR_cm_up_pca= confusion_matrix(y_test_up_pca, LR_y_pred_up_pca)
col_names = ['a_0','a_1' ]
index_names = ['p_0','p_1' ]
LR_accuracy_up_pca = (LR_cm_up_pca[0][0] + LR_cm_up_pca[1][1]) /np.sum(LR_cm_up_pca)
print(LR_accuracy_up_pca )
LR_precision_up_pca =  (LR_cm_up_pca[1][1] /np.sum(LR_cm_up_pca[: , 1])) #TP/(TP+FN) Predicted 1 and actual 1 /(total actual 1)
print (LR_precision_up_pca) 
LR_recall_up_pca = LR_cm_up_pca[1][1] /np.sum((LR_cm_up_pca[1 , :])) #TP/(TP+FN)  Predicted 1 and actual 1   / total Predicted 1
print (LR_recall_up_pca) 
#Creating the LR classification report
from sklearn.metrics import classification_report , roc_auc_score , roc_curve
print(classification_report(y_test_up_pca, LR_y_pred_up_pca)) 
#Obtaining the ROC score
roc_auc_up_pca = roc_auc_score(y_test_up_pca , LR_y_pred_up_pca)
#Obtaining false and true positives & thresholds
fpr, tpr, thresholds = roc_curve(y_test_up_pca, LR_y_pred_up_pca)
#Plotting the curve
plt.plot(fpr, tpr, label=' Logistic Regression Balanced data (after pca) (AUC = %0.03f)' % roc_auc_up_pca)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
" axis x =TP RATE = FP/ (FP +TN) = by mistake predict as 1/ total 0. low x is good"
" axis y =FP RATE = TP/(TP + FN) = said it 1 and was right /total 1. high y is good."
plt.title('ROC curve for Logistic regression upsampled data with PCA')
plt.legend(loc="upper left")
plt.show()
LR_cm_up_pca= pd.DataFrame(LR_cm_up_pca, index=col_names, columns=index_names) 
#%%
# Fitting  Regression Tree and GradientBoostin to the upsampled data
#---------------------------------------------------------------------

#Setting up pipelines with a StandardScaler function to normalize the variables
pipelines = {
    'rf' : make_pipeline(StandardScaler(), 
                         RandomForestClassifier(random_state=42, class_weight='balanced')),
    'gb' : make_pipeline(StandardScaler(), 
                         GradientBoostingClassifier(random_state=42))
}

#Setting up the "rule of thumb" hyperparameters for the Random Forest
rf_hyperparameters = {
    'randomforestclassifier__n_estimators': [100, 200],
    'randomforestclassifier__max_features': ['auto', 'sqrt', 0.33]
}

#Setting up the "rule of thumb" hyperparameters for the Gradient Boost
gb_hyperparameters = {
    'gradientboostingclassifier__n_estimators': [100, 200],
    'gradientboostingclassifier__learning_rate': [0.05, 0.1, 0.2],
    'gradientboostingclassifier__max_depth': [1, 3, 5]
}

#Creating the dictionary of hyperparameters
hyperparameters = {
    'rf' : rf_hyperparameters,
    'gb' : gb_hyperparameters
}

#Creating an empty dictionary for fitted models
up_fitted_alternative_models = {}

# Looping through model pipelines, tuning each with GridSearchCV and saving it to fitted_logreg_models
for name, pipeline in pipelines.items():
    #Creating cross-validation object from pipeline and hyperparameters
    alt_model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)
    
    #Fitting the model on X_train, y_train
    alt_model.fit(X_train_up, y_train_up)
    
    #Storing the model in fitted_logreg_models[name] 
    up_fitted_alternative_models[name] = alt_model
    
    #Printing the status of the fitting
    print(name, 'has been fitted.')

for name, model in up_fitted_alternative_models.items():
    print(name, model.best_score_ )    
    

    
RF_y_pred_up= up_fitted_alternative_models['rf'].predict(X_test_up)
print("Randon foresst summary: \n " ,  classification_report(y_test_up, RF_y_pred_up))
RF_roc_auc_up = roc_auc_score(y_test_up, RF_y_pred_up)
print("Randon foresst auc : " ,  RF_roc_auc_up )
  
GB_y_pred_up= up_fitted_alternative_models['gb'].predict(X_test_up)
print("Gradient BoostingC summary: \n " ,  classification_report(y_test_up, GB_y_pred_up))
GB_roc_auc_up = roc_auc_score(y_test_up, GB_y_pred_up)
print("Gradien tBoostingC foresst auc : " ,  GB_roc_auc_up )









