#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
warnings.filterwarnings("ignore") # Suppress warnings


# In[2]:


scaler = StandardScaler() # Initialize StandardScaler


# In[3]:


Boston = pd.read_csv('Boston.csv',index_col = 0, header=0) # Load dataset
Boston.info()


# In[15]:


### Calculate outliers
total = 0 # Initialize variable 'total' to store total outlier percentage
num_columns = len(Boston.columns) # Calculate the number of columns

for k, v in Boston.items():
    q1 = v.quantile(0.25) # Calculate first quartile
    q3 = v.quantile(0.75) # Calculate third quartile
    irq = q3 - q1 # Calculate IQR
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)] # Get outliers
    perc = np.shape(v_col)[0] * 100.0 / np.shape(Boston)[0] # Calculate percentage of outliers
    total_outlier_percentage += perc # Add to total
    print("Column %s outliers = %.2f%%" % (k, perc)) # Print percentage

average = total / num_columns # Calculate average outlier percentage for all columns
print("Overall average outlier percentage = %.2f%%" % average) # Print each column's outliers pertencange


# In[17]:


fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10)) # Set plot
index = 0
axs = axs.flatten() # Flatten the axes
for k,v in Boston.items(): 
    sns.boxplot(y=k, data=Boston, ax=axs[index]) # Create box plots for each feature
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0) # Adjust the layout


# In[21]:


fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10)) # Set plot
index = 0
axs = axs.flatten()
for k,v in Boston.items():
    sns.distplot(v, ax=axs[index]) # Create distribution plots for each feature
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0) # Adjust the layout


# In[23]:


plt.figure(figsize=(20, 10))
sns.heatmap(Boston.corr().abs(),  annot=True) # Create heatmap


# In[4]:


Boston = Boston.values # Convert DataFrame to NumPy array
Boston = scaler.fit_transform(Boston) ### Normalize input and target variables
X = Boston[:, 0:13] # Input features
y = Boston[:, 13] # Target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  # Split data into 90% train data and 10% test data


# In[98]:



def quantile_loss(y_true, y_pred, alpha): # Custom quantile loss function
    k = y_pred[:, 0] # Lower bound prediction
    m = y_pred[:, 1] # Upper bound prediction
    quantile_loss = alpha * (tf.keras.backend.maximum(0.0, y_true - m) + tf.keras.backend.maximum(0.0, k - y_true)) # Calculate quantile loss
    return quantile_loss
    

def custom_loss(y_true, y_pred, delta, alpha, l2_weight=0.01): # Final custom loss function
    huber_loss = tf.keras.losses.Huber(delta) # Huber loss
    k = y_pred[:, 0] # Lower bound prediction
    m = y_pred[:, 1] # Upper bound prediction
    lambda_value = 0.04 # Lower bound weight
    gamma_value = 0.0404 # Upper bound weight

    h_loss = huber_loss(y_true, (k + m) / 2) # Huber loss based on the average of upper bound and lower bound
    q_loss = quantile_loss(y_true, y_pred, alpha) # Quantile loss
    lower_bound_loss = tf.keras.backend.maximum(0.0, y_true - k) # Penalty for lower bound
    upper_bound_loss = tf.keras.backend.maximum(0.0, m - y_true) # Penalty for upper bound

    loss = lambda_value * tf.reduce_mean(lower_bound_loss) + gamma_value * tf.reduce_mean(upper_bound_loss) + tf.reduce_mean(q_loss) + h_loss # Total loss
    return loss
    

def create_model_with_regularization(l2_weight): # Neural network with L2 regularization
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight), input_shape=(13,)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(2)) # Output lower bound prediction and upper bound	prediction
    return model

model = create_model_with_regularization(l2_weight=0.001) # Create model
model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 0.5, 0.1)) # Compile model with custom loss function and Adam optimizer


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # Add early stopping
model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0, validation_split=0.2, callbacks=[early_stopping]) # Train the model on training data with validation split

interval_preds = model.predict(X_test) # Predict the lower and upper bounds of the prediction interval on test data
lower_preds = interval_preds[:, 0] # Lower bounds
upper_preds = interval_preds[:, 1] # Upper bounds

x_values = np.arange(len(y_test)) # X-axis values

plt.figure(figsize=(20, 6))
plt.plot(x_values, lower_preds, label='Lower Bound', color='blue') # Lower bounds
plt.plot(x_values, upper_preds, label='Upper Bound', color='red') # Upper bounds
plt.plot(x_values, y_test, label='True Value', linestyle='--', color='green') # True values

plt.xlabel('Sample Index') # X-axis label
plt.ylabel('Value') # Y-axis label
plt.legend()

plt.show()


# In[99]:



width = upper_preds - lower_preds # Calculate interval width
print("Interval width is:",width.mean())
covered_samples = ((lower_preds <= y_test) & (upper_preds >= y_test)).sum()
coverage = covered_samples / len(X_test) # # Calculate coverage probability 
print("Coverage rate is:", coverage)


# In[6]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)  # K-fold cross-validation

widths = []
coverages = []
mses = []
mapes = []
rmes = []

for train_index, val_index in kf.split(X):
    X_train_fold, X_val_fold = X[train_index], X[val_index] # Split input variables into training and validation sets for the fold
    y_train_fold, y_val_fold = y[train_index], y[val_index] # Split target variables into training and validation sets for the fold
    model = create_model_with_regularization(l2_weight=0.001) # Create model
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 0.5, 0.1)) # Compile model with custom loss function and Adam optimizer

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # Add early stopping
    model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=32, verbose=0,
              validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping]) # Train the model
    final_loss = model.evaluate(X_val_fold, y_val_fold, verbose=0) # Access the model
    print(f'Final Loss on Validation Set: {final_loss}')
    interval_preds = model.predict(X_val_fold) # Interval prediction
    lower_preds = interval_preds[:, 0] # Lower bound prediction
    upper_preds = interval_preds[:, 1] # Upper bound prediction
    
    mse = mean_squared_error(y_val_fold, (lower_preds + upper_preds) / 2) # Calculate MSE
    mape = np.mean(np.abs((y_val_fold - (lower_preds + upper_preds) / 2) / y_val_fold)) # Calculate MAPE
    rmse = np.sqrt(mse) # Calculate RMSE

    mses.append(mse)
    mapes.append(mape)
    rmes.append(rmse)

# Calculate average metrics across all folds
mean_width = np.mean(widths)
mean_coverage = np.mean(coverages)
mean_mse = np.mean(mses)
mean_mape = np.mean(mapes)
mean_rmse = np.mean(rmes)
print(f'Mean MSE: {mean_mse}, Mean MAPE: {mean_mape}, Mean RMSE: {mean_rmse}')

