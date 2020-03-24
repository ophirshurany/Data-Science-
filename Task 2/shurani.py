# import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.close('all')
# create dataframe

crime_data = pd.read_csv("Communities and Crime Data Set.csv", sep='|', encoding='utf8')
# Dropping the duplicates
crime_data = crime_data.drop_duplicates()
# view first 5 rows in df
crime_data.head()
new_col=[]
for col in crime_data.columns:
    new_col.append(col.replace(' numeric', ''))
crime_data.columns=new_col
# %%
# section 2.1, 2.2 , 2.3
# presenting all columns, number of rows and type
crime_data.info()
# feature statistics for numerical categories
crime_data.describe()
#%% -----------------------------------------------------------------------------------------
#Counting the number of missing values in the dataset¶
print("Missing values : ",crime_data.isnull().sum().sum())
#Imputing the missing values with the mean value for each column using fillna()¶
crime_data.fillna(crime_data.mean(), inplace=True)
print("Missing values after Imputing : ",crime_data.isnull().sum().sum())
#Examining the Predictors and the Response Variable Per Capita Violent Crimes
#We can see that most of the columns have minimum value of Zero which indicates a Missing value
crime_data=crime_data.drop(['state','county','community','fold'],axis=1)
#%%
corr = crime_data.corr()
f, ax = plt.subplots(figsize=(22, 22))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
cols = list(crime_data.columns)
features = cols
features.remove(crime_data.columns[-1])
X=crime_data[features]
X = pd.DataFrame(StandardScaler().fit_transform(X),columns=features)
y=crime_data[crime_data.columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#%%
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_log_error,mean_squared_error 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,roc_curve,auc
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
kfold = KFold()
grd = GradientBoostingRegressor()
#explore the hyperparameters
print(grd.get_params())
# Number of trees in random forest
n_estimators = [50,100, 200, 400]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,4,6,8,None]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
#learning rate shrinks the contribution of each tree by learning_rate.
# Create the random grid
random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf}
print(random_grid)
#search across 100 different combinations, and use all available cores
grd_random = RandomizedSearchCV(estimator = grd,
                                param_distributions = random_grid,
                                n_iter = 50, cv = kfold, verbose=2,
                                random_state=0, n_jobs = -1)

grd_best=grd_random.fit(X_train, y_train.values.ravel())
grd_best.best_params_
predictions_grd = grd_best.predict(X_test)
#%%
import shap
grd_best=grd.set_params(**grd_best.best_params_)
grd_best.fit(X_train, y_train.values.ravel())
print("Best r2 score: " +str(round(grd_best.score(X_train, y_train),3)))
# #%% -----------------------------------------------------------------------------------------
# # b) Shap value
# # -----------------------------------------------------------------------------------------
# # It is the average of the marginal contributions across all permutations.
# # -----------------------------------------------------------------------------------------
# # c) Global Interpretability
# # -----------------------------------------------------------------------------------------
# import shap

# # Build the model with the random forest regression algorithm:
# RF_best_parameters = RandomForestRegressor(random_state=24, n_estimators=100)
# RF_best_parameters.fit(X_train, y_train.values.ravel())
# shap_values = shap.TreeExplainer(RF_best_parameters).shap_values(X_train)
# shap_explainer_model = shap.TreeExplainer(RF_best_parameters)
# shap.summary_plot(shap_values, X_train, plot_type="bar")
# shap.summary_plot(shap_values, X_train)
# """ The SHAP value plot can further show the positive and negative relationships
#   of the predictors with the target variable.
#   The top 3 explainable  variable are - 
# 1.PctIlleg numeric -  percentage of kids born to never married (numeric - decimal)
# this variable  is a positively correlated with the target variable (color red).
# We were not have surprised by this result because families with married parents seems to have more
# "comfort environment" for rise children.  
# ------------------------------------------------------------------------------------------------------
# 2.PctKids2Par numeric - percentage of kids in family housing with two parents (numeric - decimal)
# this variable  is a negatively correlated with the target variable (color blue).
# we think this result is reasonable :- for Example a mother in a single-parent family ned to find time 
# for working, educate the the child and more. Unfortunately a single mom will hot have much time to \
# spend with her boy.
# ------------------------------------------------------------------------------------------------------
# 3.racePctWhite: percentage of population that is caucasian 
# Because it is the "weakest" variable among the top 3 we would make another analyses about it: 
# ------------------------------------------------------------------------------------------------------
#   """
# plt.scatter(y, df["racePctWhite numeric"], color='g')
# plt.xlabel('racePctWhite numeric (X)')
# plt.ylabel('ViolentCrimesPerPop (y)')
# plt.title(' Correlation between racePctWhite and y ')
# plt.show()
# from scipy.stats import spearmanr

# coef, p = spearmanr(y, df["racePctWhite numeric"])
# print('Spearmans correlation coefficient: %.3f' % coef)
# print('p: ', p)
# alpha = 0.05
# if p > alpha:
#     print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
# else:
#     print('Samples are correlated (reject H0) p=%.3f' % p)
# """"
# ------------------------------------------------------------------------------------------------------
# According to the SHAP values and the other tests that we have done we can see that racePctWhite can 
# effect the results(y).  
# people that are not caucasian are sometimes from low socio-economic
#   level and tend to be involed  in crime.
# ------------------------------------------------------------------------------------------------------
# """
# # d) Local Interpretability
# print("sasi0")
# random_idx = np.random.randint(len(X_test), size=3)
# X_local = X_test[random_idx, :]
# y_predicted_local = (RF_best_parameters.predict(X_local)).reshape(-1, 1)
# df_local = np.concatenate((X_local, y_predicted_local), axis=1)
# print("sasi1")

# def shap_plot(i):
#     shap_local_values_model = shap_explainer_model.shap_values(df_local)
#     p = shap.force_plot(shap_explainer_model.expected_value, shap_local_values_model[i], df_local.iloc[[i]])
#     return (p)

# print("sasi2")
# shap_plot(1)
# print("sasi3")
