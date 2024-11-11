"""
Created on Fri Nov  8 10:35:52 2024

@author: coles
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping


#loading & normalizing the dataset (step 1)
faces = fetch_olivetti_faces()
X, y = faces.data, faces.target
X /= 255.0

#splitting dataset
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=87)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=87)

#applying PCA (step 2)
pca = PCA(n_components=0.99)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

#defining autoencoder architecture (step 3)
def create_model(neurons, learning_rate, reg_strength):
    #input layer
    input_img = Input(shape=(X_train_pca.shape[1],))
    #top hidden layer 1
    x = Dense(neurons, activation='relu', kernel_regularizer=l2(reg_strength))(input_img)
    #for regularization
    #x = Dropout(0.1)(x)
    #central layer 2
    x = Dense(neurons // 4, activation='tanh', kernel_regularizer=l2(reg_strength))(x)
    #for regularization
    #x = Dropout(0.1)(x)
    #top hidden layer 3
    x = Dense(neurons, activation='relu', kernel_regularizer=l2(reg_strength))(x)
    #output layer
    decoded = Dense(X_train_pca.shape[1], activation='linear')(x)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return autoencoder

#setting up k-fold cross-val (step 3 A)
kf = KFold(n_splits=5, shuffle=True, random_state=87)
learning_rates = [0.0001, 0.001, 0.002]
reg_strengths = [1e-7]
neuron_sizes = [512, 256, 128]

best_model = None
best_params = None
lowest_val_loss = float('inf')

for neurons in neuron_sizes:
    for lr in learning_rates:
        for reg in reg_strengths:
            val_losses = []
            for train_index, val_index in kf.split(X_train_pca):
                X_fold_train, X_fold_val = X_train_pca[train_index], X_train_pca[val_index]
                model = create_model(neurons, lr, reg)
                early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
                history = model.fit(X_fold_train, X_fold_train, epochs=150, batch_size=64, verbose=0,
                                    validation_data=(X_fold_val, X_fold_val), callbacks=[early_stopping])
                val_loss = min(history.history['val_loss'])
                val_losses.append(val_loss)
            avg_val_loss = np.mean(val_losses)
            print(f'Neurons: {neurons}, LR: {lr}, Reg: {reg}, Avg Val Loss: {avg_val_loss:.4f}')
            if avg_val_loss < lowest_val_loss:
                lowest_val_loss = avg_val_loss
                best_model = model
                best_params = {'neurons': neurons, 'learning_rate': lr, 'reg_strength': reg}

#printing best params
print("Best model parameters:", best_params)

#evaluating the best model on the test set
X_test_reconstructed = best_model.predict(X_test_pca)
X_test_reconstructed = pca.inverse_transform(X_test_reconstructed)

#displaying the original and reconstructed images
plt.figure(figsize=(20, 4))
for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(X_test[i].reshape(64, 64))
    plt.gray()
    ax.axis('off')
    
    ax = plt.subplot(2, 10, i + 11)
    plt.imshow(X_test_reconstructed[i].reshape(64, 64))
    plt.gray()
    ax.axis('off')
plt.show()