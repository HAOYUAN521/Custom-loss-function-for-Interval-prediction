#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping


# In[2]:


### Generate data
np.random.seed(0) # Set random seed
num_samples = 5000 # Define total number

x1 = np.random.uniform(0, 10, num_samples) # Variable 1
x2 = np.random.uniform(0, 5, num_samples) # Variable 2
x3 = np.random.uniform(0, 2, num_samples) # Variable 3
a = 2.5 # Coefficient 1
b = 1.8 # Coefficient 2
c = 0.7 # Coefficient 3
noise = np.random.normal(0, 1, num_samples) # Set Gaussian noise
y = a * x1 + b * x2 + c * x3 + noise # Target variable
data = np.column_stack((x1, x2, x3, y)) # Covert variables to array
X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, random_state=42) # Split data into 80% train data and 20% test data

def quantile_loss(y_true, y_pred, alpha): # Custom quantile loss function
    k = y_pred[:, 0] # Lower bound prediction
    m = y_pred[:, 1] # Upper bound prediction
    quantile_loss = alpha * (tf.keras.backend.maximum(0.0, y_true - m) + tf.keras.backend.maximum(0.0, k - y_true)) # Calculate quantile loss
    return quantile_loss

def custom_loss(y_true, y_pred, delta, alpha, lambda_value, gamma_value): # Final custom loss function
    huber_loss = tf.keras.losses.Huber(delta) # Huber loss
    k = y_pred[:, 0] # Lower bound prediction
    m = y_pred[:, 1] # Upper bound prediction

    h_loss = huber_loss(y_true, (k + m) / 2) # Huber loss based on the average of upper bound and lower bound
    q_loss = quantile_loss(y_true, y_pred, alpha) # Quantile loss
    lower_bound_loss = tf.keras.backend.maximum(0.0, y_true - k) # Penalty for lower bound
    upper_bound_loss = tf.keras.backend.maximum(0.0, m - y_true) # Penalty for upper bound

    loss = lambda_value * tf.reduce_mean(lower_bound_loss) + gamma_value * tf.reduce_mean(upper_bound_loss) + tf.reduce_mean(q_loss) + h_loss # Total loss
    return loss
    
# Set different weights to upper bound and lower bound
lambda_values = [0.0001, 0.001, 0.01] # Lower bound weight
gamma_values = [0.0001, 0.001, 0.01] # Upper bound weight

plt.figure(figsize=(12, 8))

# Plot loss without early stopping
for lambda_val in lambda_values:
    for gamma_val in gamma_values:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
	    tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(2)
        ])
        model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 0.5, 0.1, lambda_val, gamma_val))
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        loss_values = history.history['loss'] # Training loss
        label_str = f'Lambda={lambda_val}, Gamma={gamma_val}' # Training plot label
        plt.plot(np.arange(1, 51), loss_values, label=label_str)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs for Different Lambda and Gamma Values')
plt.legend()
plt.show()


# In[3]:


### Plot loss with early stopping
plt.figure(figsize=(12, 8))
for lambda_val in lambda_values:
    for gamma_val in gamma_values:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(2)
        ])
        model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 0.5, 0.1, lambda_val, gamma_val))
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)
        loss_values = history.history['loss'] # Training loss
        val_loss_values = history.history['val_loss'] # Validation loss
        label_str = f'Train: Lambda={lambda_val}, Gamma={gamma_val}' # Training plot label
        plt.plot(np.arange(1, len(loss_values) + 1), loss_values, label=label_str)
        label_str = f'Validation: Lambda={lambda_val}, Gamma={gamma_val}' # Validation plot label
        plt.plot(np.arange(1, len(val_loss_values) + 1), val_loss_values, label=label_str, marker='o')
        
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss with Early Stopping')
plt.legend()
plt.show()


# In[4]:


### Generate data
np.random.seed(0)
num_samples = 1000
x1 = np.random.uniform(0, 10, num_samples)
x2 = np.random.uniform(0, 5, num_samples)
x3 = np.random.uniform(0, 2, num_samples)

a = 2.5
b = 1.8
c = 0.7

noise = np.random.normal(0, 1, num_samples)
y = a * x1 + b * x2 + c * x3 + noise

data = np.column_stack((x1, x2, x3, y))
X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, random_state=42)


# In[5]:


def quantile_loss(y_true, y_pred, alpha): # Custom quantile loss function
    k = y_pred[:, 0] # Lower bound prediction
    m = y_pred[:, 1] # Upper bound prediction
    quantile_loss = alpha * (tf.keras.backend.maximum(0.0, y_true - m) + tf.keras.backend.maximum(0.0, k - y_true)) # Calculate quantile loss
    return quantile_loss

def custom_loss(y_true, y_pred, delta, alpha, l2_weight=0.01): # Final custom loss function
    huber_loss = tf.keras.losses.Huber(delta) # Huber loss
    k = y_pred[:, 0] # Lower bound prediction
    m = y_pred[:, 1] # Upper bound prediction
    lambda_value = 0.1  # Be strict to lower bound
    gamma_value = 0.000000001
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

model = create_model_with_regularization(l2_weight=0.0001) # Create model

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



# In[6]:


width = upper_preds - lower_preds # Calculate interval width
print("Interval width is:",width.mean())
covered_samples = ((lower_preds <= y_test) & (upper_preds >= y_test)).sum()
coverage = covered_samples / len(X_test) # # Calculate coverage probability 
print("Coverage rate is:", coverage)


# In[7]:


def quantile_loss(y_true, y_pred, alpha): # Custom quantile loss function
    k = y_pred[:, 0] # Lower bound prediction
    m = y_pred[:, 1] # Upper bound prediction
    quantile_loss = alpha * (tf.keras.backend.maximum(0.0, y_true - m) + tf.keras.backend.maximum(0.0, k - y_true)) # Calculate quantile loss
    return quantile_loss

def custom_loss(y_true, y_pred, delta, alpha, l2_weight=0.01): # Final custom loss function
    huber_loss = tf.keras.losses.Huber(delta) # Huber loss
    k = y_pred[:, 0] # Lower bound prediction
    m = y_pred[:, 1] # Upper bound prediction
    lambda_value = 0.000000001
    gamma_value = 0.1  ### Be strict to upper bound
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

model = create_model_with_regularization(l2_weight=0.0001) # Create model

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



# In[8]:


width = upper_preds - lower_preds # Calculate interval width
print("Interval width is:",width.mean())
covered_samples = ((lower_preds <= y_test) & (upper_preds >= y_test)).sum()
coverage = covered_samples / len(X_test) # # Calculate coverage probability 
print("Coverage rate is:", coverage)


