import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import models, layers

# Load the Boston housing dataset
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# Feature names for Boston housing dataset
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

# Convert the data to pandas DataFrame for easy analysis
train_df = pd.DataFrame(train_data, columns=feature_names)
test_df = pd.DataFrame(test_data, columns=feature_names)

# Check for null values
print("Null values in training data:")
print(train_df.isnull().sum())

print("\nNull values in test data:")
print(test_df.isnull().sum())

# Describe the dataset
print("\nDescription of training data:")
print(train_df.describe())

# Preprocess the data using StandardScaler
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Build the neural network model
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))  # No activation function for regression

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Mean Squared Error loss for regression

# Train the model
history = model.fit(train_data, train_targets, epochs=100, batch_size=16, validation_split=0.2, verbose=0)

# Evaluate the model on the test data
test_loss, test_mae = model.evaluate(test_data, test_targets, verbose=0)

# Calculate R-squared score
predictions = model.predict(test_data)
r_squared = r2_score(test_targets, predictions)

print('Test Mean Absolute Error:', test_mae)
print('Model accuracy:', r_squared)

# Make predictions on test data
predictions = model.predict(test_data)

# Example: Print the first 10 predictions
for i in range(10):
    print("Actual Price:", test_targets[i])
    print("Predicted Price:", predictions[i][0])
    print()
