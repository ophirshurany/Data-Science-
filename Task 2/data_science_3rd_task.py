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
from sklearn.decomposition import PCA
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
X = StandardScaler().fit_transform(X)
Y=data.target_class
# Split dataset to 60% training and 40% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
#%%
# Load models
# First we reduce the data to two dimensions using PCA to capture variation
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(X)
#print(reduced_data[:10])  # print upto 10 elements
reduced_data.shape
#%%
principalDf = pd.DataFrame(data = reduced_data, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, data[['target_class']]], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [1,0]
colors = ['b', 'r']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target_class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(["Positive","Negative"])
ax.grid()
# =============================================================================
pca.explained_variance_ratio_
print("By using the attribute explained_variance_ratio_, /n "
      "you can see that the first principal component contains/n"
      ,100*round(pca.explained_variance_ratio_[0],2),"% of the variance and/n"
      " the second principal component contains/n"
      ,100*round(pca.explained_variance_ratio_[1],2),"% of the variance/n."
      "Together,the two components contain 95.80% of the information")
# =============================================================================
#%%
from sklearn import cluster
from sklearn import metrics
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
kmeans = cluster.KMeans(n_clusters=2)
clusters = kmeans.fit(reduced_data)
print(clusters)
#%%
results_KMeans=[]
results_AC_n=[]
range_n_clusters = [2, 3, 4, 5, 6,7,8]

for n_clusters in range_n_clusters:
    AC_n=AgglomerativeClustering(n_clusters=n_clusters)
    Kmeans_n = KMeans(n_clusters=n_clusters)
    clusterers=[Kmeans_n,AC_n]
    cluster_names=["KMeans","Agglomerative Clustering"]
    for clusterer in clusterers:
        cluster_labels = clusterer.fit_predict(reduced_data)
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(reduced_data, cluster_labels)
        if cluster_names[clusterers.index(clusterer)]=="KMeans":
            results_KMeans.append(silhouette_avg)
        else:
            results_AC_n.append(silhouette_avg)
fig = plt.figure()
ax = plt.axes()
ax.plot(range_n_clusters,results_KMeans,'-og', label='Kmeans')
ax.plot(range_n_clusters,results_AC_n,'-or', label='Agglomerative Clustering')
ax.set_title("The silhouette plot for the various clusters")
ax.set_ylabel("The silhouette coefficient values")
ax.set_xticks(range_n_clusters)
ax.set_xticklabels(range_n_clusters)
ax.set_xlabel("Cluster label")
ax.legend()
ax.grid(True)
#%%
ac = cluster.AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
labels = ac.fit_predict(X)
print('Cluster labels: %s' % labels)
np.shape(labels)
n_classes = 2
LABELS = [" Noise ","Pulsars"]
clf = cluster.KMeans(init='k-means++', n_clusters=2, random_state=5)
clf.fit(X_train)
print(clf.labels_.shape)
print(clf.labels_)

# Predict clusters on testing data
Y_pred = clf.predict(X_test)

from matplotlib.pyplot import cm 
n=2
c = []
color=iter(cm.rainbow(np.linspace(0,1,n)))
for i in range(n):
    c.append(next(color))


n = 2
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7.5))

pca = PCA(n_components=2)
X1 = pca.fit_transform(X)
km = KMeans(n_clusters= n , random_state=0)
y_km = km.fit_predict(X1)

for i in range(n):
    ax1.scatter(X1[y_km==i,0], X1[y_km==i,1], c=c[i], marker='o', s=40)   
ax1.set_title('K-means Clustering')


ac = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='complete')
y_ac = ac.fit_predict(X1)

for i in range(n):
    ax2.scatter(X1[y_ac==i,0], X1[y_ac==i,1], c=c[i], marker='o', s=40)
ax2.set_title('Agglomerative Clustering')


# Put a legend below current axis
plt.legend(['Negative','Positive'])
    
plt.tight_layout()
#plt.savefig('./figures/kmeans_and_ac.png', dpi=300)
plt.show()
