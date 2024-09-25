# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:21:23 2024

@author: coles
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
import numpy as np
from sklearn.datasets import fetch_openml, make_swiss_roll
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#QUESTION 1

# 1. loading the MNIST dataset.
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print(mnist.data.shape) #70,000 instances and 784 columns
X, y = mnist.data, mnist.target

# 2. displaying each digit from the dataset
fig, axes = plt.subplots(1, 10, figsize=(15, 1.5))
for i, ax in enumerate(axes):
    img = X[np.where(y == str(i))[0][0]].reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(str(i))
plt.show()

# 3. using PCA to retrieve principal components and explained variance ratio
#scaling data before pca
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#initializing pca with the first 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#outputting explained variance ratio for first 2 components
explained_variance_ratio = pca.explained_variance_ratio_
print("EVR for 1st and 2nd components:", explained_variance_ratio)

# 4. plotting projections of 1st and 2nd principal components onto 1D hyperplane
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.astype(int), cmap='viridis', alpha=0.5, edgecolor='none')
plt.colorbar(label='digit label')
plt.xlabel('1st PC')
plt.ylabel('2nd PC')
plt.title('projection of MNIST data onto first 2 principal components')
plt.show()

# 5. use incremental pca to reduce dimensionality of the mnist dataset
#initializing inc pca with 154 components
ipca = IncrementalPCA(n_components=154)
#defining batch size
batch_size = 200
#fitting incremental pca to data in batches
for X_batch in np.array_split(X_scaled, len(X_scaled) // batch_size):
    ipca.partial_fit(X_batch)
#transforming data to 154 dimensions
X_ipca = ipca.transform(X_scaled)

# 6. display the original and compressed digits from step 5
#reconstruct images from IPCA output
X_reconstructed = ipca.inverse_transform(X_ipca)

#displaying original and reconstructed images
fig, axes = plt.subplots(2, 10, figsize=(15, 3), subplot_kw={'xticks':[], 'yticks':[]})
for i in range(10):
    #original images
    axes[0, i].imshow(X[i].reshape(28, 28), cmap='gray')
    axes[0, i].set_title('Original')
    #reconstructed images
    axes[1, i].imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
    axes[1, i].set_title('Reconstructed')
plt.show()


#QUESTION 2

# 1. generating swiss roll dataset
X, color = make_swiss_roll(n_samples=1500)

# 2. plotting the generated swiss roll dataset
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.set_title('swiss roll')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


# 3. use kPCA with linear, RBF, and sigmoid kernels
#setting up kernels
kernels = ['linear', 'rbf', 'sigmoid']
kpca_results = {}
#applying kpca with different kernels
for kernel in kernels:
    kpca = KernelPCA(n_components=2, kernel=kernel, gamma=1 if kernel != 'linear' else None)
    X_kpca = kpca.fit_transform(X)
    kpca_results[kernel] = X_kpca


# 4. plotting kPCA results for linear, RBF, and sigmoid kernals
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (kernel, X_kpca) in enumerate(kpca_results.items()):
    axes[i].scatter(X_kpca[:, 0], X_kpca[:, 1], c=color, cmap=plt.cm.Spectral)
    axes[i].set_title(f'{kernel.capitalize()} Kernel')
    axes[i].set_xlabel('1st PC')
    if i == 0:
        axes[i].set_ylabel('2nd PC')
plt.show()

# 5. apply logistic regression for classification, and print best params found by GridSearchCV
#scaling data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kpca_scaler = StandardScaler()

#converting color values into categories
bins = np.linspace(min(color), max(color), num=10)
color_binned = np.digitize(color, bins)

#creating a pipeline
pipe = Pipeline([
    ('kpca', KernelPCA(n_components=2)),
    ('scale', StandardScaler()),
    ('log_reg', LogisticRegression(max_iter=10000, solver='saga'))
])
#parameter grid for GridSearchCV
param_grid = {
    'kpca__kernel': ['linear', 'rbf', 'sigmoid'],
    'kpca__gamma': np.logspace(-3, 2, 10),
    'log_reg__C': np.logspace(-4, 4, 10)
}
#setting up GridSearchCV
grid_search = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(X, color_binned)

#getting best params and best score
print("best parameters found: ", grid_search.best_params_)
print("best cross-validation accuracy: {:.2f}".format(grid_search.best_score_))


# 6. plot results from using GridSearchCV in step 5
#making a function to plot decision boundaries
def plot_decision_boundaries(X, y, model, title):
    plt.figure(figsize=(12, 9))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title(title)
    plt.xlabel('1st principal component')
    plt.ylabel('2nd principal component')
    plt.show()
    
#using best parameters from GridSearchCV
best_kpca = KernelPCA(n_components=2, kernel='sigmoid', gamma=0.001)
best_log_reg = LogisticRegression(C=1291.5496650148827, max_iter=10000, solver='saga')
#transforming data and fit model
X_kpca = best_kpca.fit_transform(X)
best_log_reg.fit(X_kpca, color_binned)
#plotting decision boundaries
plot_decision_boundaries(X_kpca, color_binned, best_log_reg, "decision boundary with best parameters")


