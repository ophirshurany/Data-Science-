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
from sklearn import cluster
from sklearn import metrics
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
kmeans = cluster.KMeans(n_clusters=2)
clusters = kmeans.fit(reduced_data)
print(clusters)
#%%
range_n_clusters = [2, 3, 4, 5, 6]
for n_clusters in range_n_clusters:
    AC_n=AgglomerativeClustering(n_clusters=n_clusters)
    Kmeans_n = KMeans(n_clusters=n_clusters)
    clusterers=[AC_n,Kmeans_n]
    cluster_names=["Agglomerative Clustering","KMeans"]
    for clusterer in clusterers:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
    
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(reduced_data) + (n_clusters + 1) * 10])
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        cluster_labels = clusterer.fit_predict(reduced_data)
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(reduced_data, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score for "+cluster_names[clusterers.index(clusterer)]+" is :", round(silhouette_avg,2)
    
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(reduced_data, cluster_labels)
    
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
   
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
    
        plt.suptitle(("Silhouette analysis for "+cluster_names[clusterers.index(clusterer)]+" on sample data ""with n_clusters = %d" % n_clusters),fontsize=14, fontweight='bold')

plt.show()
#%%
# Plot the decision boundary by building a mesh grid to populate a graph.
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1

hx = (x_max-x_min)/1000.
hy = (y_max-y_min)/1000.

xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

# Obtain labels for each point in mesh. Use last trained model.
Z = clusters.predict(np.c_[xx.ravel(), yy.ravel()])
#%%
# Find the centroids for KMeans or the cluster means for Gaussian Mixed Model

centroids = kmeans.cluster_centers_

print('*** K MEANS CENTROIDS ***')
print(centroids)

# TRANSFORM DATA BACK TO ORIGINAL SPACE FOR ANSWERING 7

print('*** CENTROIDS TRANSFERRED TO ORIGINAL SPACE ***')
print(pca.inverse_transform(centroids))

#%%
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1,figsize=(15,10))
plt.clf()
plt.imshow(Z, interpolation='Nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('Clustering on the seeds dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

#%%
ac = cluster.AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
labels = ac.fit_predict(X)
print('Cluster labels: %s' % labels)
np.shape(labels)

#%%
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
