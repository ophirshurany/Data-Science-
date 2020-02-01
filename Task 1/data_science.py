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
import warnings
warnings.filterwarnings('ignore')
plt.close('all')
import time
start_time = time.time()
#%%1. create dataframe
data = pd.read_csv("bank.csv",sep='|',encoding='utf8')
#drop duplicate data
df = data
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
    plt.figure(figsize=(12, 6))
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
#job
df.job.replace(['admin.', 'management'], 'administration_management', inplace=True)
df.loc[(df['age'] > 60 ) & (df['job'] == 'administration_management' ) , 'job'] = 'retired'
df.job.replace(['retired', 'unemployed'], 'no_active_income', inplace=True)
df.job.replace('housemaid', 'services',inplace=True)
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
df['education'] = df.education.replace(np.nan,'unknown',regex=True)
df['job'] = df.job.replace(np.nan,'unknown',regex=True)
pd.crosstab(df['job'], df['education'], rownames=['job'], colnames=['education'],margins=True)
# =============================================================================
# While imputing the values for job and education, we were cognizant of the fact that
# the correlations should make real world sense. If it didn't 
# make real world sense, we didn't replace the missing values.
# =============================================================================
#education 
#for education it makes sense to use ranking
education_dic={'illiterate': 0,'basic' : 1,'high.school' : 2,'professional.course' : 3,'university.degree' : 4}
df['education']=df.education.replace(education_dic)
#Most customers with "basic" education work as "blue-collar"
df.loc[(df['job']=='unknown') & (df['education']==1), 'job'] = 'blue-collar'
df.loc[(df['education']=='unknown') & (df['job']=='blue-collar'), 'education'] = 1
#Most customers in "services" have a "high.school"  degree
df.loc[(df['education']=='unknown') & (df['job']=='services'), 'education'] = 2
df.loc[(df['job']=='unknown') & (df['education']==2), 'job'] = 'services'
#Most customers with "professional.course education work as "technician"
df.loc[(df['job']=='unknown') & (df['education']==3), 'job'] = 'technician'
df.loc[(df['education']=='unknown') & (df['job']=='technician'), 'education'] = 3
#Most customers in "administration_management" have a "university.degree"  
df.loc[(df['education']=='unknown') & (df['job']=='administration_management'), 'education'] = 4
df.loc[(df['job']=='unknown') & (df['education']=='administration_management'), 'job'] = 'administration_management'
pd.crosstab(df['job'], df['education'], rownames=['job'], colnames=['education'],margins=True)
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
pd.crosstab(df['pdays'],df['poutcome'],dropna=False,margins=True)
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
df=df.drop(["duration","Q_4"],axis=1)
y=df.y
df=df.drop("y",axis=1)
df["y"]=y
cor = df.corr().round(1)
mask = np.zeros_like(cor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.figure(figsize=(20, 15))
heatmap=sns.heatmap(cor,mask=mask,annot=True,annot_kws={"size": 10},
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
df=df.drop(["euribor3m","emp.var.rate","poutcome_nonexistent","marital_single","pdays_missing"],axis=1)
df_copy_feature_filtered=df.copy()
#now we want to see the updated correlation matrix
cor = df.corr().round(1)
mask = np.zeros_like(cor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.figure(figsize=(20, 15))
heatmap=sns.heatmap(cor,mask=mask,annot=True,annot_kws={"size": 10},
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
scaler = MinMaxScaler(feature_range = (0,1))
normalized_df_data =scaler.fit_transform(numeric_df.values)
df_scaled=pd.DataFrame(normalized_df_data,columns=numeric_df.columns)
# =============================================================================
#%%5.1
#The optimal value for epsilon will be found at the point of maximum curvature.
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(df_scaled)
distances, indices = nbrs.kneighbors(df_scaled)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.title("Find  the  optimal "+r'$  \varepsilon$',fontsize='xx-large', fontweight='bold')
plt.ylabel("epsilon")
plt.xlabel("Feature unique values")
plt.plot([31825], [0.65], 'ro')
plt.annotate('Optimal '+r'$\varepsilon$', (31825,0.65),
            xytext=(0.8, 0.9), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=16,
            horizontalalignment='right', verticalalignment='top')
plt.tight_layout()
#eps = the best epsilon is at the "elbow" of NearestNeighbors graph
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.65,min_samples=5).fit(df_scaled)
labels=db.labels_
clusterNum=len(set(labels))
print("number of clusters is "+str(clusterNum))
noise=np.count_nonzero(labels == -1)
noise_percentage=round(100*noise/df_scaled.shape[0],0)
print("Number of outliers is "+str(noise)+ ", Noise accounts for "+str(noise_percentage)+"%  of the total dataset" )
#%%
df=df_scaled
df_DBSCAN=df.copy()
df_DBSCAN["cluster_Db"]=labels
df_DBSCAN = df_DBSCAN[df_DBSCAN.cluster_Db != -1]
df_DBSCAN=df_DBSCAN.drop("cluster_Db",axis=1)
#5.2 Lots of clusters means low number of noise, therefore low number of outliers.
#5.3 - Another method to remove outliers - LOF
from sklearn.neighbors import LocalOutlierFactor
# fit the model for outlier detection (default)
LOF = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
y_pred = LOF.fit_predict(df_scaled)
X_scores = LOF.negative_outlier_factor_
LOF_outliers=np.count_nonzero(y_pred == -1)
LOF_outliers_percentage=round(100*np.count_nonzero(y_pred == -1)/df_scaled.shape[0],0)
print("Number of outliers by LOF is "+str(LOF_outliers)+ ", Noise accounts for "+str(LOF_outliers_percentage)+"%  of the total dataset" )
plt.close('all')
#%% 6. Predictive Models
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
from sklearn.utils import resample
kfold = model_selection.KFold(n_splits=10, random_state=42)
y=df_DBSCAN.y
X=df_DBSCAN.drop("y",axis=1)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0) #80/20 split
#%%Upsample
y_train_no_yes=y_train.value_counts()
df_majority = df_DBSCAN[df_DBSCAN.y==0]
df_minority = df_DBSCAN[df_DBSCAN.y==1]
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=y_train_no_yes[0],    # to match majority class
                                 random_state=123) # reproducible results
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df_upsampled.y.value_counts()
y_upsample=df_upsampled.y
X_upsample=df_upsampled.drop("y",axis=1)
x_train_up, x_test_up, y_train_up, y_test_up = model_selection.train_test_split(X_upsample, y_upsample, test_size=0.2, random_state=0) #80/20 split
x_train=x_train_up
y_train= y_train_up
#%% 6.1.1 Random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
#explore the hyperparameters
pprint(rfc.get_params())
# Number of trees in random forest
n_estimators = [500]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 9, num = 8)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rfc,
                               param_distributions = random_grid,
                               n_iter = 100, cv = kfold, verbose=2,
                               random_state=42, n_jobs = -1)
rf_random.fit(x_train, y_train)
rf_random.best_params_
rf_best_random = rf_random.best_estimator_
prediction_RF = rf_best_random.predict(x_test)
print("for Random Forest we get " +str(round(accuracy_score(y_test, prediction_RF),5)))

CM_RF=confusion_matrix(y_test, prediction_RF)
df_cm = pd.DataFrame(CM_RF, index = ["Predicted No","Predicted Yes"],
                  columns = ["Actual No","Actual Yes"])
plt.figure()
sns.heatmap(df_cm,cmap="bwr", annot=True)
print(classification_report(y_test, prediction_RF,target_names=["no","yes"]))
#AUC
probs_RF = rf_random.predict_proba(x_test)
preds_RF = probs_RF[:,1]
fprrfc, tprrfc, thresholdrfc = metrics.roc_curve(y_test, preds_RF)
roc_aucrfc = metrics.auc(fprrfc, tprrfc)
#%% 6.1.2. ADABOOST
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ADA = AdaBoostClassifier()
#explore the hyperparameters
pprint(ADA.get_params())
#learning rate shrinks the contribution of each tree by learning_rate.
learning_rate = [round(float(x),2) for x in np.linspace(start = 0.2, stop = 2, num = 10)]
#algorithm ===================================================================
# If ‘SAMME.R’ then use the SAMME.R real boosting algorithm.
# base_estimator must support calculation of class probabilities.
# If ‘SAMME’ then use the SAMME discrete boosting algorithm.
# The SAMME.R algorithm typically converges faster than SAMME,
# achieving a lower test error with fewer boosting iterations.
# =============================================================================
algorithm = ["SAMME", "SAMME.R"]
n_estimators = [500]
#The base estimator from which the boosted ensemble is built
base_estimator= [DecisionTreeClassifier(max_depth=x) for x in np.linspace(2, 8, num = 3)]
base_estimator.append(None)
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'algorithm': algorithm,
               'base_estimator': base_estimator, 
               'learning_rate':learning_rate}
pprint(random_grid)
# search across 100 different combinations, and use all available cores
ADA_random = RandomizedSearchCV(estimator = ADA,
                               param_distributions = random_grid,
                               n_iter = 100, cv = kfold, verbose=2,
                               random_state=42, n_jobs = -1)
ADA_random.fit(x_train, y_train)
ADA_random.best_params_
ADA_best_random = ADA_random.best_estimator_
predictions_ADA = ADA_best_random.predict(x_test)
print("for ADABOOST we get " +str(round(accuracy_score(y_test, predictions_ADA),5)))
CM_ADA=confusion_matrix(y_test, predictions_ADA)
df_cm = pd.DataFrame(CM_ADA, index = ["Predicted No","Predicted Yes"],
                  columns = ["Actual No","Actual Yes"])
plt.figure()
sns.heatmap(df_cm,cmap=cmap, annot=True)
print(classification_report(y_test, predictions_ADA,target_names=["no","yes"]))
#AUC
probs_ADA = ADA_random.predict_proba(x_test)
preds_ADA = probs_ADA[:,1]
fprADA, tprADA, thresholdADA = metrics.roc_curve(y_test, preds_ADA)
roc_aucADA = metrics.auc(fprADA, tprADA)
#%% 6.1.3.  Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
grd = GradientBoostingClassifier()
#explore the hyperparameters
pprint(grd.get_params())
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 500, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt',"log2"]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 10, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
#learning rate shrinks the contribution of each tree by learning_rate.
#learning_rate=[round(float(x),2) for x in np.linspace(start = 0, stop = 0.1, num = 11)]
#loss function to be optimized
loss=["deviance", "exponential"]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'loss' : loss}
 #               'learning_rate': learning_rate}
pprint(random_grid)
#search across 100 different combinations, and use all available cores
grd_random = RandomizedSearchCV(estimator = grd,
                                param_distributions = random_grid,
                                n_iter = 100, cv = kfold, verbose=2,
                                random_state=42, n_jobs = -1)
grd_random.fit(x_train, y_train)
grd_random.best_params_
grd_best_random = grd_random.best_estimator_
predictions_grd = grd_best_random.predict(x_test)
print("for Gradient Boosting we get " +str(round(accuracy_score(y_test, predictions_grd),5)))
CM_grd=confusion_matrix(y_test, predictions_grd)
df_cm = pd.DataFrame(CM_grd, index = ["Predicted No","Predicted Yes"],
                  columns = ["Actual No","Actual Yes"])
plt.figure()
sns.heatmap(df_cm,cmap=cmap, annot=True)
print(classification_report(y_test, predictions_grd,target_names=["no","yes"]))
#AUC
probs_grd = grd_random.predict_proba(x_test)
preds_grd = probs_grd[:,1]
fprgrd, tprgrd, thresholdgrd = metrics.roc_curve(y_test, preds_grd)
roc_aucgrd = metrics.auc(fprgrd, tprgrd)
#%%AUC Curve
sns.set()
plt.figure()
plt.plot([0, 1], [0, 1],'r--')
plt.title('Predictive models RUC Comparison',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.xlabel('False Positive Rate',fontsize=15)
plt.plot(fprgrd, tprgrd, label = 'Gradient Boosting AUC = %0.3f' % roc_aucgrd)
plt.plot(fprrfc, tprrfc, label = 'Random Forest AUC = %0.3f' % roc_aucrfc)
plt.plot(fprADA, tprADA, label = 'ADABOOST AUC = %0.3f' % roc_aucADA)
plt.legend(loc = 'lower right', prop={'size': 16})
print("--- %s minutes ---" % (round(time.time()/60 - start_time/60,2)))