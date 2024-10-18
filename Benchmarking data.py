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


# In[7]:


Concrete = pd.read_csv('Concrete_Data.csv',index_col = 0, header=0)
Concrete.info()


# In[4]:


### Calculate outliers
total_outlier_percentage = 0
num_columns = len(Concrete.columns)

for k, v in Concrete.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(Concrete)[0]
    total_outlier_percentage += perc
    print("Column %s outliers = %.2f%%" % (k, perc))

average_outlier_percentage = total_outlier_percentage / num_columns
print("Overall average outlier percentage = %.2f%%" % average_outlier_percentage)


# In[5]:


fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in Concrete.items():
    sns.boxplot(y=k, data=Concrete, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[6]:


fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in Concrete.items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[29]:


plt.figure(figsize=(20, 10))
sns.heatmap(Concrete.corr().abs(),  annot=True)


# In[8]:


Concrete = Concrete.values
Concrete = scaler.fit_transform(Concrete) ### Normalize input and target variables
X = Concrete[:, 0:7]
y = Concrete[:, 7]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  ### Split data into 90% train data and 10% test data


# In[ ]:


### Custom quantile loss function
def quantile_loss(y_true, y_pred, alpha):
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    quantile_loss = alpha * (tf.keras.backend.maximum(0.0, y_true - m) + tf.keras.backend.maximum(0.0, k - y_true))
    return quantile_loss
    
### Final custom loss function
def custom_loss(y_true, y_pred, delta, alpha):
    huber_loss = tf.keras.losses.Huber(delta)
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    lambda_value = 0.03902
    gamma_value = 0.040502
    h_loss = huber_loss(y_true, (k + m) / 2)
    q_loss = quantile_loss(y_true, y_pred, alpha)
    lower_bound_loss = tf.keras.backend.maximum(0.0, y_true - k)
    upper_bound_loss = tf.keras.backend.maximum(0.0, m - y_true)

    loss = lambda_value * tf.reduce_mean(lower_bound_loss) + gamma_value * tf.reduce_mean(upper_bound_loss) + tf.reduce_mean(q_loss) + h_loss
    return loss

### Neural network with l2 Regularization
def create_model_with_regularization(l2_weight):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight), input_shape=(7,)))
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


# In[ ]:


### Calculate interval width and coverage probability
width = upper_preds - lower_preds
print("Interval width is:",width.mean())
covered_samples = ((lower_preds <= y_test) & (upper_preds >= y_test)).sum()
coverage = covered_samples / len(X_test)
print("Coverage rate is:", coverage)


# In[9]:


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


# In[10]:


Energy = pd.read_excel('Energy.xlsx', index_col=None, header=None)
Energy.info()


# In[3]:


### Calculate outliers
total_outlier_percentage = 0
num_columns = len(Energy.columns)

for k, v in Energy.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(Energy)[0]
    total_outlier_percentage += perc
    print("Column %s outliers = %.2f%%" % (k, perc))

average_outlier_percentage = total_outlier_percentage / num_columns
print("Overall average outlier percentage = %.2f%%" % average_outlier_percentage)


# In[7]:


fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in Energy.items():
    sns.boxplot(y=k, data=Energy, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[5]:


fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in Energy.items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[8]:


plt.figure(figsize=(20, 10))
sns.heatmap(Energy.corr().abs(),  annot=True)


# In[11]:


Energy = Energy.values
Energy = scaler.fit_transform(Energy)  ### Normalize input and target variables
X = Energy[:, 0:8]
y = Energy[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  ### Split data into 90% train data and 10% test data


# In[ ]:


### Custom quantile loss function
def quantile_loss(y_true, y_pred, alpha):
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    quantile_loss = alpha * (tf.keras.backend.maximum(0.0, y_true - m) + tf.keras.backend.maximum(0.0, k - y_true))
    return quantile_loss

### Final custom loss function
def custom_loss(y_true, y_pred, delta, alpha):
    huber_loss = tf.keras.losses.Huber(delta)
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    lambda_value = 0.0629
    gamma_value = 0.0789
    h_loss = huber_loss(y_true, (k + m) / 2)
    q_loss = quantile_loss(y_true, y_pred, alpha)
    lower_bound_loss = tf.keras.backend.maximum(0.0, y_true - k)
    upper_bound_loss = tf.keras.backend.maximum(0.0, m - y_true)

    loss = lambda_value * tf.reduce_mean(lower_bound_loss) + gamma_value * tf.reduce_mean(upper_bound_loss) + tf.reduce_mean(q_loss) + h_loss
    return loss
    
### Neural network with l2 Regularization
def create_model_with_regularization(l2_weight):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight), input_shape=(8,)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(2))
    return model

model = create_model_with_regularization(l2_weight=0.00001)
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


# In[ ]:


### Calculate interval width and coverage probability
width = upper_preds - lower_preds
print("Interval width is:",width.mean())
covered_samples = ((lower_preds <= y_test) & (upper_preds >= y_test)).sum()
coverage = covered_samples / len(X_test)
print("Coverage rate is:", coverage)


# In[12]:


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
    model = create_model_with_regularization(l2_weight=0.00001)
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


# In[15]:


Wine = pd.read_csv('winequality-red.csv')
Wine.info()


# In[10]:


### Calculate outliers
total_outlier_percentage = 0
num_columns = len(Energy.columns)

for k, v in Wine.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(Wine)[0]
    total_outlier_percentage += perc
    print("Column %s outliers = %.2f%%" % (k, perc))

average_outlier_percentage = total_outlier_percentage / num_columns
print("Overall average outlier percentage = %.2f%%" % average_outlier_percentage)


# In[11]:


fig, axs = plt.subplots(ncols=6, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in Wine.items():
    sns.boxplot(y=k, data=Wine, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[12]:


fig, axs = plt.subplots(ncols=6, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in Wine.items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[13]:


plt.figure(figsize=(20, 10))
sns.heatmap(Wine.corr().abs(),  annot=True)


# In[16]:


Wine = Wine.values
Wine = scaler.fit_transform(Wine) ### Normalize input and target variables
X = Wine[:, 0:11]
y = Wine[:, 11]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)   ### Split data into 90% train data and 10% test data


# In[ ]:


### Custom quantile loss function
def quantile_loss(y_true, y_pred, alpha):
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    quantile_loss = alpha * (tf.keras.backend.maximum(0.0, y_true - m) + tf.keras.backend.maximum(0.0, k - y_true))
    return quantile_loss

### Final custom loss function
def custom_loss(y_true, y_pred, delta, alpha):
    huber_loss = tf.keras.losses.Huber(delta)
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    lambda_value = 0.025
    gamma_value = 0.017
    h_loss = huber_loss(y_true, (k + m) / 2)
    q_loss = quantile_loss(y_true, y_pred, alpha)
    lower_bound_loss = tf.keras.backend.maximum(0.0, y_true - k)
    upper_bound_loss = tf.keras.backend.maximum(0.0, m - y_true)

    loss = lambda_value * tf.reduce_mean(lower_bound_loss) + gamma_value * tf.reduce_mean(upper_bound_loss) + tf.reduce_mean(q_loss) + h_loss
    return loss

### Neural network with l2 Regularization
def create_model_with_regularization(l2_weight):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight), input_shape=(11,)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(2))
    return model

model = create_model_with_regularization(l2_weight=0.001)
model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 0.65, 0.1))

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


# In[ ]:


### Calculate interval width and coverage probability
width = upper_preds - lower_preds
print("Interval width is:",width.mean())
covered_samples = ((lower_preds <= y_test) & (upper_preds >= y_test)).sum()
coverage = covered_samples / len(X_test)
print("Coverage rate is:", coverage)


# In[17]:


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
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 0.65, 0.1))
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


# In[18]:


Kin8nm = pd.read_csv('dataset_2175_kin8nm.csv', index_col=None, header=None)
Kin8nm.info()


# In[16]:


### Calculate outliers
total_outlier_percentage = 0
num_columns = len(Energy.columns)

for k, v in Kin8nm.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(Kin8nm)[0]
    total_outlier_percentage += perc
    print("Column %s outliers = %.2f%%" % (k, perc))

average_outlier_percentage = total_outlier_percentage / num_columns
print("Overall average outlier percentage = %.2f%%" % average_outlier_percentage)


# In[23]:


fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in Kin8nm.items():
    sns.boxplot(y=k, data=Kin8nm, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[24]:


fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in Kin8nm.items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[19]:


plt.figure(figsize=(20, 10))
sns.heatmap(Kin8nm.corr().abs(),  annot=True)


# In[19]:


Kin8nm = Kin8nm.values
Kin8nm = scaler.fit_transform(Kin8nm)  ### Normalize input and target variables
X = Kin8nm[:, 0:8]
y = Kin8nm[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42) ### Split data into 90% train data and 10% test data


# In[ ]:


### Custom quantile loss function
def quantile_loss(y_true, y_pred, alpha):
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    quantile_loss = alpha * (tf.keras.backend.maximum(0.0, y_true - m) + tf.keras.backend.maximum(0.0, k - y_true))
    return quantile_loss

### Final custom loss function
def custom_loss(y_true, y_pred, delta, alpha):
    huber_loss = tf.keras.losses.Huber(delta)
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    lambda_value = 0.048989
    gamma_value = 0.048989
    h_loss = huber_loss(y_true, (k + m) / 2)
    q_loss = quantile_loss(y_true, y_pred, alpha)
    lower_bound_loss = tf.keras.backend.maximum(0.0, y_true - k)
    upper_bound_loss = tf.keras.backend.maximum(0.0, m - y_true)

    loss = lambda_value * tf.reduce_mean(lower_bound_loss) + gamma_value * tf.reduce_mean(upper_bound_loss) + tf.reduce_mean(q_loss) + h_loss
    return loss

### Neural network with l2 Regularization
def create_model_with_regularization(l2_weight):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight), input_shape=(8,)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(2))
    return model

model = create_model_with_regularization(l2_weight=0.0001)
model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 0.4, 0.1))

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


# In[ ]:


### Calculate interval width and coverage probability
width = upper_preds - lower_preds
print("Interval width is:",width.mean())
covered_samples = ((lower_preds <= y_test) & (upper_preds >= y_test)).sum()
coverage = covered_samples / len(X_test)
print("Coverage rate is:", coverage)


# In[20]:


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
    model = create_model_with_regularization(l2_weight=0.0001)
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 0.4, 0.1))
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


# In[21]:


powerplant = pd.read_excel('powerplant.xlsx', index_col=None, header=None)
powerplant.info()


# In[11]:


### Calculate outliers
total_outlier_percentage = 0
num_columns = len(powerplant.columns)

for k, v in powerplant.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(powerplant)[0]
    total_outlier_percentage += perc
    print("Column %s outliers = %.2f%%" % (k, perc))

average_outlier_percentage = total_outlier_percentage / num_columns
print("Overall average outlier percentage = %.2f%%" % average_outlier_percentage)


# In[13]:


fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in powerplant.items():
    sns.boxplot(y=k, data=powerplant, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[14]:


fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in powerplant.items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[15]:


plt.figure(figsize=(20, 10))
sns.heatmap(powerplant.corr().abs(),  annot=True)


# In[22]:


powerplant = powerplant.values
powerplant = scaler.fit_transform(powerplant)  ### Normalize input and target variables
X = powerplant[:, 0:4]
y = powerplant[:, 4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  ### Split data into 90% train data and 10% test data


# In[ ]:


### Custom quantile loss function
def quantile_loss(y_true, y_pred, alpha):
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    quantile_loss = alpha * (tf.keras.backend.maximum(0.0, y_true - m) + tf.keras.backend.maximum(0.0, k - y_true))
    return quantile_loss

### Final custom loss function
def custom_loss(y_true, y_pred, delta, alpha):
    huber_loss = tf.keras.losses.Huber(delta)
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    lambda_value = 0.052
    gamma_value = 0.055
    h_loss = huber_loss(y_true, (k + m) / 2)
    q_loss = quantile_loss(y_true, y_pred, alpha)
    lower_bound_loss = tf.keras.backend.maximum(0.0, y_true - k)
    upper_bound_loss = tf.keras.backend.maximum(0.0, m - y_true)

    loss = lambda_value * tf.reduce_mean(lower_bound_loss) + gamma_value * tf.reduce_mean(upper_bound_loss) + tf.reduce_mean(q_loss) + h_loss
    return loss

### Neural network with l2 Regularization
def create_model_with_regularization(l2_weight):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight), input_shape=(4,)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(2))
    return model

model = create_model_with_regularization(l2_weight=0.0001)
model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 0.4, 0.1))

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


# In[ ]:


### Calculate interval width and coverage probability
width = upper_preds - lower_preds
print("Interval width is:",width.mean())
covered_samples = ((lower_preds <= y_test) & (upper_preds >= y_test)).sum()
coverage = covered_samples / len(X_test)
print("Coverage rate is:", coverage)


# In[23]:


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
    model = create_model_with_regularization(l2_weight=0.0001)
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 0.4, 0.1))
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


# In[24]:


Naval = pd.read_csv('Naval.csv',index_col = 0, header=0)
Naval.info()


# In[3]:


### Calculate outliers
total_outlier_percentage = 0
num_columns = len(Naval.columns)

for k, v in Naval.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(Naval)[0]
    total_outlier_percentage += perc
    print("Column %s outliers = %.2f%%" % (k, perc))

average_outlier_percentage = total_outlier_percentage / num_columns
print("Overall average outlier percentage = %.2f%%" % average_outlier_percentage)


# In[5]:


fig, axs = plt.subplots(ncols=9, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in Naval.items():
    sns.boxplot(y=k, data=Naval, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[6]:


fig, axs = plt.subplots(ncols=9, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in Naval.items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[8]:


plt.figure(figsize=(20, 10))
sns.heatmap(Naval.corr().abs(),  annot=True)


# In[25]:


Naval= Naval.values
Naval = scaler.fit_transform(Naval)  ### Normalize input and target variables
X = Naval[:, 0:16]
y = Naval[:, 16]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)   ### Split data into 90% train data and 10% test data


# In[ ]:


### Custom quantile loss function
def quantile_loss(y_true, y_pred, alpha):
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    quantile_loss = alpha * (tf.keras.backend.maximum(0.0, y_true - m) + tf.keras.backend.maximum(0.0, k - y_true))
    return quantile_loss

### Final custom loss function
def custom_loss(y_true, y_pred, delta, alpha):
    huber_loss = tf.keras.losses.Huber(delta)
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    lambda_value = 0.082
    gamma_value = 0.061
    h_loss = huber_loss(y_true, (k + m) / 2)
    q_loss = quantile_loss(y_true, y_pred, alpha)
    lower_bound_loss = tf.keras.backend.maximum(0.0, y_true - k)
    upper_bound_loss = tf.keras.backend.maximum(0.0, m - y_true)

    loss = lambda_value * tf.reduce_mean(lower_bound_loss) + gamma_value * tf.reduce_mean(upper_bound_loss) + tf.reduce_mean(q_loss) + h_loss
    return loss

### Neural network with l2 Regularization
def create_model_with_regularization(l2_weight):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight), input_shape=(16,)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(2))
    return model

model = create_model_with_regularization(l2_weight=0.00001)
model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 0.4, 0.1))

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


# In[ ]:


### Calculate interval width and coverage probability
width = upper_preds - lower_preds
print("Interval width is:",width.mean())
covered_samples = ((lower_preds <= y_test) & (upper_preds >= y_test)).sum()
coverage = covered_samples / len(X_test)
print("Coverage rate is:", coverage)


# In[26]:


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
    model = create_model_with_regularization(l2_weight=0.00001)
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 0.4, 0.1))
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


# In[27]:


Yacht = pd.read_csv('Yacht.csv',index_col = 0, header=0)
Yacht.info()


# In[4]:


### Calculate outliers
total_outlier_percentage = 0
num_columns = len(Yacht.columns)

for k, v in Yacht.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(Yacht)[0]
    total_outlier_percentage += perc
    print("Column %s outliers = %.2f%%" % (k, perc))

average_outlier_percentage = total_outlier_percentage / num_columns
print("Overall average outlier percentage = %.2f%%" % average_outlier_percentage)


# In[5]:


fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in Yacht.items():
    sns.boxplot(y=k, data=Yacht, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[6]:


fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in Yacht.items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[7]:


plt.figure(figsize=(20, 10))
sns.heatmap(Yacht.corr().abs(),  annot=True)


# In[28]:


Yacht = Yacht.values
Yacht = scaler.fit_transform(Yacht)  ### Normalize input and target variables
X = Yacht[:, 0:6]
y = Yacht[:, 6]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)   ### Split data into 90% train data and 10% test data


# In[ ]:


### Custom quantile loss function
def quantile_loss(y_true, y_pred, alpha):
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    quantile_loss = alpha * (tf.keras.backend.maximum(0.0, y_true - m) + tf.keras.backend.maximum(0.0, k - y_true))
    return quantile_loss

### Final custom loss function
def custom_loss(y_true, y_pred, delta, alpha):
    huber_loss = tf.keras.losses.Huber(delta)
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    lambda_value = 0.035
    gamma_value = 0.0851
    h_loss = huber_loss(y_true, (k + m) / 2)
    q_loss = quantile_loss(y_true, y_pred, alpha)
    lower_bound_loss = tf.keras.backend.maximum(0.0, y_true - k)
    upper_bound_loss = tf.keras.backend.maximum(0.0, m - y_true)

    loss = lambda_value * tf.reduce_mean(lower_bound_loss) + gamma_value * tf.reduce_mean(upper_bound_loss) + tf.reduce_mean(q_loss) + h_loss
    return loss

### Neural network with l2 Regularization
def create_model_with_regularization(l2_weight):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight), input_shape=(6,)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(2))
    return model

model = create_model_with_regularization(l2_weight=0.0001)
model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 1, 0.1))

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


# In[ ]:


### Calculate interval width and coverage probability
width = upper_preds - lower_preds
print("Interval width is:",width.mean())
covered_samples = ((lower_preds <= y_test) & (upper_preds >= y_test)).sum()
coverage = covered_samples / len(X_test)
print("Coverage rate is:", coverage)


# In[29]:


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
    model = create_model_with_regularization(l2_weight=0.0001)
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 1, 0.1))
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


# In[41]:


features = ['year', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29', 't30', 't31', 't32', 't33', 't34', 't35', 't36 ', 't37', 't38', 't39', 't40', 't41', 't42', 't43', 't44', 't45', 't46', 't47', 't48', 't49', 't50', 't51', 't52', 't53' , 't54', 't55', 't56', 't57', 't58', 't59', 't60', 't61', 't62', 't63', 't64', 't65', 't66', 't67', 't68', 't69', 't70', 't71', 't72', 't73', 't74', 't75', 't76', 't77', 't78', 't79', 't80', 't81', 't82', 't83', 't84', 't85', 't86', 't87', 't88', 't89', 't90']
MSD = pd.read_csv('YearPredictionMSD.csv', names=features)
MSD.info()


# In[9]:


### Calculate outliers
total_outlier_percentage = 0
num_columns = len(MSD.columns)

for k, v in MSD.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(MSD)[0]
    total_outlier_percentage += perc
    print("Column %s outliers = %.2f%%" % (k, perc))

average_outlier_percentage = total_outlier_percentage / num_columns
print("Overall average outlier percentage = %.2f%%" % average_outlier_percentage)


# In[42]:


fig, axs = plt.subplots(ncols=6, nrows=16, figsize=(20, 40))
index = 0
axs = axs.flatten()
for k,v in MSD.items():
    sns.boxplot(y=k, data=MSD, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[43]:


fig, axs = plt.subplots(ncols=6, nrows=16, figsize=(20, 40))
index = 0
axs = axs.flatten()
for k,v in MSD.items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[8]:


plt.figure(figsize=(20, 10))
sns.heatmap(MSD.corr().abs(),  annot=True)


# In[31]:


MSD = MSD.values
MSD = scaler.fit_transform(MSD)  ### Normalize input and target variables
X = MSD[:, 1:]
y = MSD[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)   ### Split data into 90% train data and 10% test data


# In[ ]:


### Custom quantile loss function
def quantile_loss(y_true, y_pred, alpha):
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    quantile_loss = alpha * (tf.keras.backend.maximum(0.0, y_true - m) + tf.keras.backend.maximum(0.0, k - y_true))
    return quantile_loss

### Final custom loss function
def custom_loss(y_true, y_pred, delta, alpha):
    huber_loss = tf.keras.losses.Huber(delta)
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    lambda_value = 0.000001
    gamma_value = 0.019
    h_loss = huber_loss(y_true, (k + m) / 2)
    q_loss = quantile_loss(y_true, y_pred, alpha)
    lower_bound_loss = tf.keras.backend.maximum(0.0, y_true - k)
    upper_bound_loss = tf.keras.backend.maximum(0.0, m - y_true)

    loss = lambda_value * tf.reduce_mean(lower_bound_loss) + gamma_value * tf.reduce_mean(upper_bound_loss) + tf.reduce_mean(q_loss) + h_loss
    return loss

### Neural network with l2 Regularization
def create_model_with_regularization(l2_weight):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight), input_shape=(90,)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(2))
    return model

model = create_model_with_regularization(l2_weight=0.00001)
model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 0.8, 0.1))

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


# In[ ]:


### Calculate interval width and coverage probability
width = upper_preds - lower_preds
print("Interval width is:",width.mean())
covered_samples = ((lower_preds <= y_test) & (upper_preds >= y_test)).sum()
coverage = covered_samples / len(X_test)
print("Coverage rate is:", coverage)


# In[32]:


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
    model = create_model_with_regularization(l2_weight=0.00001)
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 0.8, 0.1))
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


# In[33]:


Protein = pd.read_csv('protein.csv',index_col=None, header=None)
Protein.info()


# In[11]:


### Calculate outliers
total_outlier_percentage = 0
num_columns = len(Protein.columns)

for k, v in Protein.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(Protein)[0]
    total_outlier_percentage += perc
    print("Column %s outliers = %.2f%%" % (k, perc))

average_outlier_percentage = total_outlier_percentage / num_columns
print("Overall average outlier percentage = %.2f%%" % average_outlier_percentage)


# In[12]:


fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in Protein.items():
    sns.boxplot(y=k, data=Protein, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[13]:


fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in Protein.items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[14]:


plt.figure(figsize=(20, 10))
sns.heatmap(Protein.corr().abs(),  annot=True)


# In[34]:


Protein = Protein.values
Protein = scaler.fit_transform(Protein)  ### Normalize input and target variables
X = Protein[:, 1:]
y = Protein[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)   ### Split data into 90% train data and 10% test data


# In[ ]:


### Custom quantile loss function
def quantile_loss(y_true, y_pred, alpha):
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    quantile_loss = alpha * (tf.keras.backend.maximum(0.0, y_true - m) + tf.keras.backend.maximum(0.0, k - y_true))
    return quantile_loss

### Final custom loss function
def custom_loss(y_true, y_pred, delta, alpha):
    huber_loss = tf.keras.losses.Huber(delta)
    k = y_pred[:, 0]
    m = y_pred[:, 1]
    lambda_value = 0.0399
    gamma_value = 0.000000000001
    h_loss = huber_loss(y_true, (k + m) / 2)
    q_loss = quantile_loss(y_true, y_pred, alpha)
    lower_bound_loss = tf.keras.backend.maximum(0.0, y_true - k)
    upper_bound_loss = tf.keras.backend.maximum(0.0, m - y_true)

    loss = lambda_value * tf.reduce_mean(lower_bound_loss) + gamma_value * tf.reduce_mean(upper_bound_loss) + tf.reduce_mean(q_loss) + h_loss
    return loss

### Neural network with l2 Regularization
def create_model_with_regularization(l2_weight):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight), input_shape=(9,)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_weight)))
    model.add(tf.keras.layers.Dense(2))
    return model

model = create_model_with_regularization(l2_weight=0.00001)
model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 0.9, 0.1))

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


# In[ ]:


### Calculate interval width and coverage probability
width = upper_preds - lower_preds
print("Interval width is:",width.mean())
covered_samples = ((lower_preds <= y_test) & (upper_preds >= y_test)).sum()
coverage = covered_samples / len(X_test)
print("Coverage rate is:", coverage)


# In[36]:


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
    model = create_model_with_regularization(l2_weight=0.00001)
    model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 0.9, 0.1))
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

