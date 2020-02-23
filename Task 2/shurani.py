# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
# create dataframe

df = pd.read_csv("Communities and Crime Data Set.csv", sep='|', encoding='utf8')
# Dropping the duplicates
df = df.drop_duplicates()
# view first 5 rows in df
df.head()
# %%
# section 2.1, 2.2 , 2.3
# presenting all columns, number of rows and type
df.info()
# feature statistics for numerical categories
df.describe()
#%% -----------------------------------------------------------------------------------------
# task 2 - preprocessing"
# handling  missing data
nan_rows = df.isnull().sum(axis=1)
sum_nan_rows = sum(nan_rows)  # Present total rows with NA.
print(sum_nan_rows)
nan_col = len(df) - df.count()
nan_col_per = (nan_col / len(df)) * 100  # Present NAn % in each feature.
print(nan_col_per)
# county numeric
# acording to the Attribute Information - not predictive, and many missing values (numeric)
df = df.drop(['county numeric'], axis=1)
nan_col = len(df) - df.count()
nan_col_per = (nan_col / len(df)) * 100  # Present NAn % in each feature.
print(nan_col_per)
# community numeric
# vcommunity: numeric code for community - not predictive and many missing values (numeric)"
df = df.drop(['community numeric'], axis=1)
nan_col = len(df) - df.count()
nan_col_per = (nan_col / len(df)) * 100  # Present NAn % in each feature.
print(nan_col_per)
# OtherPerCap numeric
print(df['OtherPerCap numeric'])
#%%
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(df[["OtherPerCap numeric"]])
df["OtherPerCap numeric"] = imputer.transform(df[["OtherPerCap numeric"]]).ravel()
# the other columns with missing values have more than 84% missing value in each column.
# therefore, we can't trust these features
nan_col = len(df) - df.count()
nan_col_per = (nan_col / len(df)) * 100  # Present NAn % in each feature.
print(nan_col_per)

df = df.dropna(axis='columns')
# df after fixing missing data:
print(sum(df.isnull().sum(axis=1)))  # 0 rows with NA.
#%% -----------------------------------------------------------------------------------------
# task 3 - SHAP"
# -----------------------------------------------------------------------------------------
# a) predictive model
# -----------------------------------------------------------------------------------------
# separate  dataset to train and test
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train_before_sv = X_train
# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train = pd.DataFrame(X_train, columns=X_train_before_sv.columns)
#%% -----------------------------------------------------------------------------------------
# creating predective model using cv - Random Forest
# -----------------------------------------------------------------------------------------
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
max_depths = [1,2,4,6,8,None]
rf = RandomForestRegressor()
# Setting up the "rule of thumb" hyperparameters for the Random Forest
rf_hyperparameters = {
     'n_estimators': [50,100, 200, 400],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10],
     'max_depth': max_depths
}
kfold =KFold()
# Creating an empty dictionary for fitted models
#rf_cv = RandomizedSearchCV(estimator=rf, param_distributions=rf_hyperparameters , cv=10, n_jobs=-1)
rf_cv = RandomizedSearchCV(estimator = rf,
                               param_distributions = rf_hyperparameters,
                               n_iter = 50, cv = kfold, verbose=2,
                               random_state=42, n_jobs = -1)
# Fitting the model on X_train, y_train
rf_cv.fit(X_train, y_train.values.ravel())
print( "rf has been fitted.")
print("rf score:  ", rf_cv.best_score_ )
print("rf pars")
print(rf_cv.best_params_ )
y_pred_GB = rf_cv.predict(X_test)
#%% -----------------------------------------------------------------------------------------
# b) Shap value
# -----------------------------------------------------------------------------------------
# It is the average of the marginal contributions across all permutations.
# -----------------------------------------------------------------------------------------
# c) Global Interpretability
# -----------------------------------------------------------------------------------------
import shap

# Build the model with the random forest regression algorithm:
RF_best_parameters = RandomForestRegressor(random_state=24, n_estimators=100)
RF_best_parameters.fit(X_train, y_train.values.ravel())
shap_values = shap.TreeExplainer(RF_best_parameters).shap_values(X_train)
shap_explainer_model = shap.TreeExplainer(RF_best_parameters)
shap.summary_plot(shap_values, X_train, plot_type="bar")
shap.summary_plot(shap_values, X_train)
""" The SHAP value plot can further show the positive and negative relationships
 of the predictors with the target variable.
 The top 3 explainable  variable are - 
1.PctIlleg numeric -  percentage of kids born to never married (numeric - decimal)
this variable  is a positively correlated with the target variable (color red).
We were not have surprised by this result because families with married parents seems to have more
"comfort environment" for rise children.  
------------------------------------------------------------------------------------------------------
2.PctKids2Par numeric - percentage of kids in family housing with two parents (numeric - decimal)
this variable  is a negatively correlated with the target variable (color blue).
we think this result is reasonable :- for Example a mother in a single-parent family ned to find time 
for working, educate the the child and more. Unfortunately a single mom will hot have much time to \
spend with her boy.
------------------------------------------------------------------------------------------------------
3.racePctWhite: percentage of population that is caucasian 
Because it is the "weakest" variable among the top 3 we would make another analyses about it: 
------------------------------------------------------------------------------------------------------
 """
plt.scatter(y, df["racePctWhite numeric"], color='g')
plt.xlabel('racePctWhite numeric (X)')
plt.ylabel('ViolentCrimesPerPop (y)')
plt.title(' Correlation between racePctWhite and y ')
plt.show()
from scipy.stats import spearmanr

coef, p = spearmanr(y, df["racePctWhite numeric"])
print('Spearmans correlation coefficient: %.3f' % coef)
print('p: ', p)
alpha = 0.05
if p > alpha:
    print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
else:
    print('Samples are correlated (reject H0) p=%.3f' % p)
""""
------------------------------------------------------------------------------------------------------
According to the SHAP values and the other tests that we have done we can see that racePctWhite can 
effect the results(y).  
people that are not caucasian are sometimes from low socio-economic
 level and tend to be involed  in crime.
------------------------------------------------------------------------------------------------------
"""
# d) Local Interpretability
print("sasi0")
random_idx = np.random.randint(len(X_test), size=3)
X_local = X_test[random_idx, :]
y_predicted_local = (RF_best_parameters.predict(X_local)).reshape(-1, 1)
df_local = np.concatenate((X_local, y_predicted_local), axis=1)
print("sasi1")

def shap_plot(i):
    shap_local_values_model = shap_explainer_model.shap_values(df_local)
    p = shap.force_plot(shap_explainer_model.expected_value, shap_local_values_model[i], df_local.iloc[[i]])
    return (p)

print("sasi2")
shap_plot(1)
print("sasi3")
