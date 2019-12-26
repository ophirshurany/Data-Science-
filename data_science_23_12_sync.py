# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 23:51:46 2019
@author: OphirShurany
"""
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
scaler = MinMaxScaler((-1,1))
normalized_df_data =scaler.fit_transform(numeric_df.values)
x_scaled=numeric_df
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
plt.plot([32758], [2.8], 'ro')
plt.annotate('Optimal '+r'$\varepsilon$', (32758,2.8),
            xytext=(0.8, 0.9), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=16,
            horizontalalignment='right', verticalalignment='top')
plt.tight_layout()
#eps = the best epsilon is at the "elbow" of NearestNeighbors graph
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=2.8).fit(x_scaled)

labels=db.labels_
clusterNum=len(set(labels))
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
#%%
"""# Applying PCA
from sklearn.decomposition import PCA
pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
#choose  vars than variance more than 0.06
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
pca = PCA(2)
#X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)
#explained_variance = pca.explained_variance_ratio_, """
#%%
# Fitting Logistic Regression to the Training set
#pca
X = df.drop("y",axis=1)
y = df.y.values
nonZeroYCount = np.count_nonzero(df[28])
prop_nonzero = nonZeroYCount/len(df[28])
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

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
print(classification_report(y_test, LR_y_pred)) 
#Obtaining the ROC score
roc_auc = roc_auc_score(y_test, LR_y_pred)
#Obtaining false and true positives & thresholds
fpr, tpr, thresholds = roc_curve(y_test, LR_y_pred)
#Plotting the curve
plt.plot(fpr, tpr, label=' Logistic Regression (AUC = %0.03f)' % roc_auc)
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
LR_cm= pd.DataFrame(LR_cm, index=col_names, columns=index_names) 
# checking another therold for the Logistic Regression
THRESHOLD = 0
classifier.fit(X_train, y_train)
# Predicting the Test set results
for THRESHOLD in [x * 0.05 for x in range(2, 20)]:
    LR_y_pred = classifier.predict(X_test)
    LR_y_pred= np.where(classifier.predict_proba(X_test)[:,1] > THRESHOLD, 1, 0)
    print("AUC IS :", round(roc_auc_score(y_test, LR_y_pred) , 3) , "for this thersold: " , round(THRESHOLD,2))
LR_y_pred_T_15= np.where(classifier.predict_proba(X_test)[:,1] >0.15, 1, 0)  
LR_cm_T_15= confusion_matrix(y_test, LR_y_pred_T_15)
LR_cm_T_15= pd.DataFrame(LR_cm_T_15, index=col_names, columns=index_names) 
print(classification_report(y_test, LR_y_pred_T_15)) 
LR_y_pred = classifier.predict(X_test)
print(classification_report(y_test, LR_y_pred))     
#Obtaining the ROC score
roc_auc = roc_auc_score(y_test, LR_y_pred_T_15)
#Obtaining false and true positives & thresholds
fpr, tpr, thresholds = roc_curve(y_test, LR_y_pred_T_15)
#Plotting the curve
plt.plot(fpr, tpr, label=' Logistic Regression (AUC = %0.03f)' % roc_auc)
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
"""# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show() """
#%%
# Fitting  Regression Tree to the Training set
#---------------------------------------------------------------------
# Fitting Random Forest Classification to the Training set
#Setting up pipelines with a StandardScaler function to normalize the variables
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
"""
pipelines = {

 }
#Setting up the "rule of thumb" hyperparameters for the Random Forest
pipelines = {
    'rf' : make_pipeline(StandardScaler(), 
                         RandomForestClassifier(random_state=42, class_weight='balanced')),
  
}

#Setting up the "rule of thumb" hyperparameters for the Random Forest
rf_hyperparameters = {
    'randomforestclassifier__n_estimators': [100, 200],
    'randomforestclassifier__max_features': ['auto', 'sqrt', 0.33]
}


#Creating the dictionary of hyperparameters
hyperparameters = {
    'rf' : rf_hyperparameters
}

#Creating an empty dictionary for fitted models
fitted_alternative_models = {}
i = 0
# Looping through model pipelines, tuning each with GridSearchCV and saving it to fitted_logreg_models
for name, pipeline in pipelines.items():
    i = i +1
    print(i)
    #Creating cross-validation object from pipeline and hyperparameters
    alt_model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)
        #Fitting the model on X_train, y_train
    alt_model.fit(X_train, y_train)
        #Storing the model in fitted_logreg_models[name] 
    fitted_alternative_models[name] = alt_model
        #Printing the status of the fitting
    print(name, 'has been fitted.')
#Displaying the best_score_ for each fitted model
for name, model in fitted_alternative_models.items():
    print("best"  , name, model.best_score_ )"""
    


#%% KNN model
from sklearn.neighbors import KNeighborsClassifier
# search for an optimal value of K for KNN
# list of integers 1 to50
# integers we want to try
k_range = range(1,10)

# list of scores from k_range
k_scores = []

# 1. we will loop through reasonable values of k
"""for k in k_range:
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
plt.ylabel('Cross-Validated Accuracy') """
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
#Separating the imbalanced observations into 2 separate datasets
from sklearn.utils import resample
df_majority = df[df[25]==0]
df_minority = df[df[25]==1]
#Upsampling the minority class
df_minority_upsampled = resample(df_minority, replace=True, n_samples=30066, random_state=42)
#Concatenating two datasets
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
#New class counts
df_upsampled[25].value_counts()
#%%
#try logistic regression om the balanced data
X_b = df_upsampled.iloc[:, 0:25].values
y_b = df_upsampled.iloc[:, 25].values
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_b, y_b, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_b, y_train_b)
# Predicting the Test set results
LR_y_pred_b = classifier.predict(X_test_b)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
LR_cm_b= confusion_matrix(y_test_b, LR_y_pred_b)
col_names = ['a_0','a_1' ]
index_names = ['p_0','p_1' ]
LR_accuracy_b = (LR_cm_b[0][0] + LR_cm_b[1][1]) /np.sum(LR_cm_b)
print(LR_accuracy_b )
LR_precision_b =  (LR_cm_b[1][1] /np.sum(LR_cm_b[: , 1])) #TP/(TP+FN) Predicted 1 and actual 1 /(total actual 1)
print (LR_precision_b) 
LR_recall_b = LR_cm_b[1][1] /np.sum((LR_cm_b[1 , :])) #TP/(TP+FN)  Predicted 1 and actual 1   / total Predicted 1
print (LR_recall_b) 
#Creating the LR classification report
from sklearn.metrics import classification_report , roc_auc_score , roc_curve
print(classification_report(y_test_b, LR_y_pred_b)) 
#Obtaining the ROC score
roc_auc_b = roc_auc_score(y_test_b , LR_y_pred_b)
#Obtaining false and true positives & thresholds
fpr, tpr, thresholds = roc_curve(y_test_b, LR_y_pred_b)
#Plotting the curve
plt.plot(fpr, tpr, label=' Logistic Regression Balanced data (AUC = %0.03f)' % roc_auc_b)
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
LR_cm_b= pd.DataFrame(LR_cm_b, index=col_names, columns=index_names)
#%%
#trying pca on the balanced logistic regression
from sklearn.decomposition import PCA
X_train_b_pca, X_test_b_pca, y_train_b_pca, y_test_b_pca = train_test_split(X_b, y_b, test_size = 0.2, random_state = 0)
pca_b= PCA()
X_train_b_pca = pca_b.fit_transform(X_train_b_pca)
X_test_b_pca = pca_b.transform(X_test_b_pca)
explained_variance_pca_b =pca_b.explained_variance_ratio_
#choose  vars than variance more than 0.06
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
pca_b = PCA(8)
X_train_b_pca = pca_b.fit_transform(X_train_b_pca)
X_test_b_pca = pca_b.transform(X_test_b_pca)
explained_variance_pca_b =pca_b.explained_variance_ratio_
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_b_pca, y_train_b_pca)
# Predicting the Test set results
LR_y_pred_b_pca = classifier.predict(X_test_b_pca)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
LR_cm_b_pca= confusion_matrix(y_test_b_pca, LR_y_pred_b_pca)
col_names = ['a_0','a_1' ]
index_names = ['p_0','p_1' ]
LR_accuracy_b_pca = (LR_cm_b_pca[0][0] + LR_cm_b_pca[1][1]) /np.sum(LR_cm_b_pca)
print(LR_accuracy_b_pca )
LR_precision_b_pca =  (LR_cm_b_pca[1][1] /np.sum(LR_cm_b_pca[: , 1])) #TP/(TP+FN) Predicted 1 and actual 1 /(total actual 1)
print (LR_precision_b_pca) 
LR_recall_b_pca = LR_cm_b_pca[1][1] /np.sum((LR_cm_b_pca[1 , :])) #TP/(TP+FN)  Predicted 1 and actual 1   / total Predicted 1
print (LR_recall_b_pca) 
#Creating the LR classification report
from sklearn.metrics import classification_report , roc_auc_score , roc_curve
print(classification_report(y_test_b_pca, LR_y_pred_b_pca)) 
#Obtaining the ROC score
roc_auc_b_pca = roc_auc_score(y_test_b_pca , LR_y_pred_b_pca)
#Obtaining false and true positives & thresholds
fpr, tpr, thresholds = roc_curve(y_test_b_pca, LR_y_pred_b_pca)
#Plotting the curve
plt.plot(fpr, tpr, label=' Logistic Regression Balanced data (after pca) (AUC = %0.03f)' % roc_auc_b_pca)
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
LR_cm_b_pca= pd.DataFrame(LR_cm_b_pca, index=col_names, columns=index_names) 
#trying pca on the balanced logistic regression


















"""# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show() """
  
  




  
  
  
  
  





#%%

"""shurani 20.12                                                
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
    sns.countplot(x=col,hue='y',data=df) """
