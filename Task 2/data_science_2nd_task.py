"""
Created on Fri Jan 31 18:22:18 2020
@author: Ophir Shurany
"""
#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import  train_test_split
import warnings
warnings.filterwarnings('ignore')
plt.close('all')
import time
start_time = time.time()
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc,fbeta_score
from sklearn.metrics import f1_score
sns.set()
#%% Create dataframe
data = pd.read_csv("pulsar_stars.csv")
#view first 5 rows in df
#%%
cols = list(data.columns)
features = cols
features.remove('target_class')
# Normalization
X=data[features]
Y=data.target_class
X=StandardScaler().fit_transform(X)
#%%
# Load models

model = KNeighborsClassifier(n_neighbors=2)
RANDOM_STATE=0

class DummySampler:

    def sample(self, X, y):
        return X, y
    def fit(self, X, y):
        return self
    def fit_resample(self, X, y):
        return self.sample(X, y)
# prepare configuration for cross validation test harness
seed = 7
# prepare models
RUS=RandomUnderSampler()
ADASYN=ADASYN()
ROS= RandomOverSampler()
SMOTE=SMOTE()
Combine=SMOTETomek()
Samplers = []
Samplers.append(('Original', DummySampler()))
Samplers.append(('RUS', RUS))
Samplers.append(('ROS', ROS))
Samplers.append(('ADASYN', ADASYN))
Samplers.append(('SMOTE', SMOTE))
Samplers.append(('Combine', Combine))
# evaluate each model in turn
score = ['accuracy', 'f1',"precision","recall"] #Different measures
for scoring in score:
    results = []
    names = []
    predictions=[]
    f1_score_tot=[];f2_score_tot=[]
    fpr_tot=[];tpr_tot=[];roc_auc_tot=[]
    for name, sampler in Samplers:
        X_sampled, Y_sampled = sampler.fit_resample(X,Y)
        print("The number of samples in ",name," dataset is" ,X_sampled.shape[0])
        print(Y_sampled.value_counts())
        print("Ratio samples in ",name," dataset target values is",round(Y_sampled.value_counts()[1]/Y_sampled.value_counts()[0],3))

        X_train_test, X_val, Y_train_test, Y_val =train_test_split(X_sampled, Y_sampled, test_size=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_train_test, Y_train_test, test_size=0.2, random_state=42)
        cv_results = cross_val_score(model, X_val, Y_val,cv=10, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
        model.fit(X_train, y_train)
        y_pred=model.predict(X_test)
        predictions.append(y_pred)
        probs= model.predict_proba(X_test)
        preds = probs[:,1]
        #F2 score
        f1_measure=f1_score(y_test, y_pred)
        f1_score_tot.append(f1_measure)
        f2_score=fbeta_score(y_test, y_pred, beta=2)
        f2_score_tot.append(f2_score)
        #ROC-AUC
        fpr, tpr, threshold = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        fpr_tot.append(fpr);tpr_tot.append(tpr);roc_auc_tot.append(roc_auc)
        print(classification_report(y_test, model.predict(X_test),target_names=["no","yes"]))
        CM=confusion_matrix(y_test, y_pred)
        print(pd.DataFrame(CM, index = ["Predicted No","Predicted Yes"],
                  columns = ["Actual No","Actual Yes"]))
    # boxplot algorithm comparison
    fig1 = plt.figure()
    fig1.suptitle('Samplers Comparison by '+scoring)
    ax1 = fig1.add_subplot(111)
    plt.boxplot(results)
    ax1.set_xticklabels(names)
    ax1.set_ylabel(scoring+" [%]")

plt.figure()
plt.plot([0, 1], [0, 1],'r--')
plt.title('Predictive models RUC Comparison',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.xlabel('False Positive Rate',fontsize=15)
for i in range(len(names)):
    plt.plot(fpr_tot[i], tpr_tot[i], label = names[i]+' AUC = %0.3f' % roc_auc_tot[i])
    plt.legend(loc = 'lower right', prop={'size': 16})  

df = pd.DataFrame(list(zip(f1_score_tot,f2_score_tot)),columns =['F1 measure', 'F2 measure'],index=names) 
print("--- %s minutes ---" % (round(time.time()/60 - start_time/60,2)))
