import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load training and testing data
X_train = np.load("Data/processed/X_train.npy")
y_train = np.load("Data/processed/y_train.npy")
X_test = np.load("Data/processed/X_test.npy")
y_test = np.load("Data/processed/y_test.npy")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Build LSTM model
model = Sequential([
    LSTM(64, input_shape=(10, 2)),
    Dense(32, activation="relu"),
    Dense(2)
])

# Compile model
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# Show model structure
model.summary()

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate on test data
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

print("\n===== MODEL RESULTS =====")
print("Test Loss:", test_loss)
print("Test MAE:", test_mae)

# Save model
model.save("models/cyclone_lstm.keras")

print("\nModel saved successfully!")
print("models/cyclone_lstm.keras")