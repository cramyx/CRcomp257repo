# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:06:28 2024

@author: coles
"""
#COMP257 A2

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

#QUESTION 1

# 1. retrieve & load olivetti faces dataset
data = fetch_olivetti_faces()
X = data.images
y = data.target

# 2. split training set, validation set, & test set using stratified sampling
# reshaping to 2d array
X = X.reshape((len(X), -1))

# stratified split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=87)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.18, stratify=y_train_val, random_state=87)

print("train set:", X_train.shape, y_train.shape)
print("validation set:", X_val.shape, y_val.shape)
print("test set:", X_test.shape, y_test.shape)

# 3. training a classifier using k-fold cross validation to predict person in images
# normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# classifier
svm = SVC(kernel='linear', random_state=87)

# set up k-fold cross val
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=87)

cv_scores = []
for train_index, test_index in kfold.split(X_train_scaled, y_train):
    #splitting into kfold
    X_fold_train, X_fold_test = X_train_scaled[train_index], X_train_scaled[test_index]
    y_fold_train, y_fold_test = y_train[train_index], y_train[test_index]
    
    #training model
    svm.fit(X_fold_train, y_fold_train)
    
    #evaluating model
    fold_pred = svm.predict(X_fold_test)
    fold_accuracy = accuracy_score(y_fold_test, fold_pred)
    cv_scores.append(fold_accuracy)

# cross val scores
print("cross validation accuracy scores:", cv_scores)
print("mean cross validation accuracy:", np.mean(cv_scores))

# training the final model & evaluating validation set
svm.fit(X_train_scaled, y_train)
val_predictions = svm.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, val_predictions)
print("validation set accuracy:", val_accuracy)

# 4. use k-means to reduce dimensionality
#range to try
k_range = range(2, 15)
#storing silhouette score and k variables
best_silhouette = -1
best_k = 0
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=87)
    cluster_labels = kmeans.fit_predict(X_train_scaled)
    silhouette_avg = silhouette_score(X_train_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    
    #printing silhouette scores for each k
    print("for n_clusters =", k, " average silhouette_score is :", silhouette_avg)
    
    #checking if silhouette score is best
    if silhouette_avg > best_silhouette:
        best_silhouette = silhouette_avg
        best_k = k

#printing best k and its silhouette score
print("best silhouette score:", best_silhouette, "found for k =", best_k)

#using best k to perform k-means
kmeans = KMeans(n_clusters=best_k, random_state=87)
X_train_kmeans = kmeans.fit_transform(X_train_scaled)
X_val_kmeans = kmeans.transform(X_val_scaled)

#re-training svm with k-means reduced dataset
svm.fit(X_train_kmeans, y_train)
val_predictions_kmeans = svm.predict(X_val_kmeans)
val_accuracy_kmeans = accuracy_score(y_val, val_predictions_kmeans)
print("validation set accuracy after k-means dimensionality reduction:", val_accuracy_kmeans)

# 5. using set from step 4 to train a classifier as in step 3

# seting up k-fold cross-val again
cv_scores_kmeans = []
for train_index, test_index in kfold.split(X_train_kmeans, y_train):
    #splitting data into k-fold
    X_kfold_train, X_kfold_test = X_train_kmeans[train_index], X_train_kmeans[test_index]
    y_kfold_train, y_kfold_test = y_train[train_index], y_train[test_index]
    
    #training model on the reduced data
    svm.fit(X_kfold_train, y_kfold_train)
    
    #evaluating the model on the k-fold test set
    kfold_pred = svm.predict(X_kfold_test)
    kfold_accuracy = accuracy_score(y_kfold_test, kfold_pred)
    cv_scores_kmeans.append(kfold_accuracy)

#printing cross-val results for k-means reduced data
print("k-means reduced data cross-validation accuracy scores:", cv_scores_kmeans)
print("mean cross-validation accuracy on K-Means reduced data:", np.mean(cv_scores_kmeans))

#training final model on full k-means reduced training data & evaluating on the reduced validation set
svm.fit(X_train_kmeans, y_train)
val_predictions_kmeans_final = svm.predict(X_val_kmeans)
val_accuracy_kmeans_final = accuracy_score(y_val, val_predictions_kmeans_final)
print("Validation set accuracy on K-Means reduced data:", val_accuracy_kmeans_final)

# 6. applying DBSCAN for clustering the entire dataset
# generating k-distance graph for DBSCAN eps parameter tuning
X_scaled = scaler.fit_transform(X)
neighbors = NearestNeighbors(n_neighbors=4)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances, axis=0)
distances = distances[:, 3]

#plotting k-distance graph
plt.plot(distances)
plt.title('k-distance graph for DBSCAN')
plt.xlabel('points sorted by distance')
plt.ylabel('4th nearest neighbor distance')
plt.show()
eps_value = distances[300] + 0.1

#setting min_samples to a higher value for more robust clustering
min_samples_value = 8

#applying DBSCAN with adjusted parameters based on k-distance graph
dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value, metric='euclidean')
dbscan_clusters = dbscan.fit_predict(X_scaled)

#analyzing results of DBSCAN with new params
print("DBSCAN cluster labels:", dbscan_clusters)
unique_dbscan_clusters = np.unique(dbscan_clusters)
print("unique clusters found by DBSCAN:", unique_dbscan_clusters)
noise_points_count = np.sum(dbscan_clusters == -1)
print("number of noise points detected by DBSCAN:", noise_points_count)

#checking if DBSCAN resulted in any valid clusters
if len(unique_dbscan_clusters) > 1:
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_clusters, cmap='viridis', marker='o', s=5)
    plt.title('DBSCAN Clustering Results')
    plt.xlabel('scaled feature 1')
    plt.ylabel('scaled feature 2')
    plt.colorbar(label='cluster label')
    plt.show()
else:
    print("all data points are considered noise by DBSCAN with the current settings.")




