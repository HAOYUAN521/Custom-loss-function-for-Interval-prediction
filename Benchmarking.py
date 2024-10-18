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
warnings.filterwarnings("ignore")


# In[2]:


scaler = StandardScaler()


# In[3]:


Boston = pd.read_csv('Boston.csv',index_col = 0, header=0)
Boston.info()


# In[15]:


### Calculate outliers
total_outlier_percentage = 0
num_columns = len(Boston.columns)

for k, v in Boston.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(Boston)[0]
    total_outlier_percentage += perc
    print("Column %s outliers = %.2f%%" % (k, perc))

average_outlier_percentage = total_outlier_percentage / num_columns
print("Overall average outlier percentage = %.2f%%" % average_outlier_percentage)


# In[17]:


fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in Boston.items():
    sns.boxplot(y=k, data=Boston, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[21]:


fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in Boston.items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[23]:


plt.figure(figsize=(20, 10))
sns.heatmap(Boston.corr().abs(),  annot=True)


# In[4]:


Boston = Boston.values
Boston = scaler.fit_transform(Boston) ### Normalize input and target variables
X = Boston[:, 0:13]
y = Boston[:, 13]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  ### Split data into 90% train data and 10% test data


# In[98]:


### Custom quantile loss function
def quantile_loss(y_true, y_pred, alpha):
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    quantile_loss = alpha * (tf.keras.backend.maximum(0.0, y_true - m) + tf.keras.backend.maximum(0.0, k - y_true))
    return quantile_loss
    
### Final custom loss function
def custom_loss(y_true, y_pred, delta, alpha, l2_weight=0.01):
    huber_loss = tf.keras.losses.Huber(delta)
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    lambda_value = 0.04
    gamma_value = 0.0404

    h_loss = huber_loss(y_true, (k + m) / 2)
    q_loss = quantile_loss(y_true, y_pred, alpha)
    lower_bound_loss = tf.keras.backend.maximum(0.0, y_true - k)
    upper_bound_loss = tf.keras.backend.maximum(0.0, m - y_true)

    loss = lambda_value * tf.reduce_mean(lower_bound_loss) + gamma_value * tf.reduce_mean(upper_bound_loss) + tf.reduce_mean(q_loss) + h_loss
    return loss
    
### Neural network with l2 Regularization
def create_model_with_regularization(l2_weight):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight), input_shape=(13,)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(2))
    return model

model = create_model_with_regularization(l2_weight=0.001)
model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 0.5, 0.1))

### Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0, validation_split=0.2, callbacks=[early_stopping])

interval_preds = model.predict(X_test)
lower_preds = interval_preds[:, 0]
upper_preds = interval_preds[:, 1]

x_values = np.arange(len(y_test))

plt.figure(figsize=(20, 6))
plt.plot(x_values, lower_preds, label='Lower Bound', color='blue')
plt.plot(x_values, upper_preds, label='Upper Bound', color='red')
plt.plot(x_values, y_test, label='True Value', linestyle='--', color='green')

plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()

plt.show()


# In[99]:


### Calculate interval width and coverage probability
width = upper_preds - lower_preds
print("Interval width is:",width.mean())
covered_samples = ((lower_preds <= y_test) & (upper_preds >= y_test)).sum()
coverage = covered_samples / len(X_test)
print("Coverage rate is:", coverage)


# In[6]:


### Calculate regression metrics using K-folder
kf = KFold(n_splits=5, shuffle=True, random_state=42)  

widths = []
coverages = []
mses = []
mapes = []
rmes = []

for train_index, val_index in kf.split(X):
    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]
    model = create_model_with_regularization(l2_weight=0.001)
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 0.5, 0.1))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=32, verbose=0,
              validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping])
    final_loss = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    print(f'Final Loss on Validation Set: {final_loss}')
    interval_preds = model.predict(X_val_fold)
    lower_preds = interval_preds[:, 0]
    upper_preds = interval_preds[:, 1]
    
    mse = mean_squared_error(y_val_fold, (lower_preds + upper_preds) / 2)
    mape = np.mean(np.abs((y_val_fold - (lower_preds + upper_preds) / 2) / y_val_fold))
    rmse = np.sqrt(mse)

    mses.append(mse)
    mapes.append(mape)
    rmes.append(rmse)

mean_width = np.mean(widths)
mean_coverage = np.mean(coverages)
mean_mse = np.mean(mses)
mean_mape = np.mean(mapes)
mean_rmse = np.mean(rmes)
print(f'Mean MSE: {mean_mse}, Mean MAPE: {mean_mape}, Mean RMSE: {mean_rmse}')

