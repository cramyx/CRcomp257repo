# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:31:53 2024

@author: coles
"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import minkowski
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)

#step 1
#loading dataset
faces = fetch_olivetti_faces()
X, y = faces.data, faces.target

#step 2
#stratified split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=87)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=87)

#step 3
#k-fold cross-val with a classifier
clf = KNeighborsClassifier(n_neighbors=5)
kf = StratifiedKFold(n_splits=5)

#evaluate classifier on training set
scores = cross_val_score(clf, X_train, y_train, cv=kf)
print("Initial Average accuracy:", np.mean(scores))

#step 4
#hierarchical clustering with different distance measures
#euclidean
cluster_euclidean = AgglomerativeClustering(linkage='ward', affinity='euclidean', n_clusters=100)
labels_euclidean = cluster_euclidean.fit_predict(X_train)

#minkowski
pairwise_minkowski = np.zeros((X_train.shape[0], X_train.shape[0]))
for i in range(X_train.shape[0]):
    for j in range(i, X_train.shape[0]):
        distance = minkowski(X_train[i], X_train[j], p=3)
        pairwise_minkowski[i, j] = distance
        pairwise_minkowski[j, i] = distance

cluster_minkowski = AgglomerativeClustering(linkage='average', affinity='precomputed', n_clusters=100)
labels_minkowski = cluster_minkowski.fit_predict(pairwise_minkowski)

#cosine
pairwise_cosine_similarity = 1 - np.dot(X_train, X_train.T) / (np.linalg.norm(X_train, axis=1).reshape(-1, 1) * np.linalg.norm(X_train, axis=1).reshape(1, -1))
pairwise_cosine_distance = 1 - (1 - pairwise_cosine_similarity) / 2
np.fill_diagonal(pairwise_cosine_distance, 0)

cluster_cosine = AgglomerativeClustering(linkage='average', affinity='precomputed', n_clusters=100)
labels_cosine = cluster_cosine.fit_predict(pairwise_cosine_distance)

#step 5
#silhouette scores for cluster eval
silhouette_euclidean = silhouette_score(X_train, labels_euclidean, metric='euclidean')
silhouette_minkowski = silhouette_score(pairwise_minkowski, labels_minkowski, metric='precomputed')
silhouette_cosine = silhouette_score(pairwise_cosine_distance, labels_cosine, metric='precomputed')

print("Silhouette Scores (100 clusters):")
print("Euclidean:", silhouette_euclidean)
print("Minkowski:", silhouette_minkowski)
print("Cosine:", silhouette_cosine)

#determining optimal number of clusters for each measure
cluster_range = range(2, 100)
silhouette_scores_euclidean = []
silhouette_scores_minkowski = []
silhouette_scores_cosine = []

#euclidean
for n_clusters in cluster_range:
    cluster_euclidean = AgglomerativeClustering(linkage='ward', affinity='euclidean', n_clusters=n_clusters)
    labels = cluster_euclidean.fit_predict(X_train)
    silhouette_scores_euclidean.append(silhouette_score(X_train, labels, metric='euclidean'))

#minkowski
for n_clusters in cluster_range:
    cluster_minkowski = AgglomerativeClustering(linkage='average', affinity='precomputed', n_clusters=n_clusters)
    labels = cluster_minkowski.fit_predict(pairwise_minkowski)
    silhouette_scores_minkowski.append(silhouette_score(pairwise_minkowski, labels, metric='precomputed'))

#cosine
for n_clusters in cluster_range:
    cluster_cosine = AgglomerativeClustering(linkage='average', affinity='precomputed', n_clusters=n_clusters)
    labels = cluster_cosine.fit_predict(pairwise_cosine_distance)
    silhouette_scores_cosine.append(silhouette_score(pairwise_cosine_distance, labels, metric='precomputed'))

#plotting silhouette scores
plt.figure(figsize=(12, 6))
plt.plot(cluster_range, silhouette_scores_euclidean, label='Euclidean')
plt.plot(cluster_range, silhouette_scores_minkowski, label='Minkowski')
plt.plot(cluster_range, silhouette_scores_cosine, label='Cosine')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Distance Measures')
plt.legend()
plt.show()

#step 6 
#choosing optimal # of clusters and re-evaluating
#determining optimal # of clusters for euclidean distance from step 5
optimal_clusters = 45

#re-run the clustering for euclidean with the optimal cluster #
cluster_euclidean_optimal = AgglomerativeClustering(linkage='ward', affinity='euclidean', n_clusters=optimal_clusters)
labels_euclidean_optimal = cluster_euclidean_optimal.fit_predict(X_train)
X_train_reduced_optimal = labels_euclidean_optimal.reshape(-1, 1)

#re-evaluate classifier on the optimally reduced data
scores_reduced_optimal = cross_val_score(clf, X_train_reduced_optimal, y_train, cv=kf)
print("Reduced Data Average accuracy with optimal clusters(45):", np.mean(scores_reduced_optimal))