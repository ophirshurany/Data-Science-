from sklearn.model_selection import train_test_split
import lightgbm as lgb
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
# print the JS visualization code to the notebook
shap.initjs()

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
cols = list(crime_data.columns)
features = cols
features.remove(crime_data.columns[-1])
X=crime_data[features]
#X = StandardScaler().fit_transform(X)
y=crime_data[crime_data.columns[-1]]
#%%
# create a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
d_train = lgb.Dataset(X_train, label=y_train)
d_test = lgb.Dataset(X_test, label=y_test)
X_test_array = X_test.toarray()
params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 10,
    "verbose": -1,
    "min_data": 100,
    "boost_from_average": True
}

model = lgb.train(params, d_train, 10000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=1000)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)



#shap.force_plot(explainer.expected_value, shap_values[:1000,:], X_display.iloc[:1000,:])

shap.summary_plot(shap_values, X)
shap.summary_plot(shap_values, X_test)
