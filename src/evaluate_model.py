import numpy as np
import tensorflow as tf

# Load test data
X_test = np.load("Data/processed/X_test.npy")
y_test = np.load("Data/processed/y_test.npy")

# Load trained model
model = tf.keras.models.load_model("models/cyclone_lstm.keras")

# Make predictions
predictions = model.predict(X_test, verbose=0)

# Calculate coordinate errors
lat_error = predictions[:, 0] - y_test[:, 0]
lon_error = predictions[:, 1] - y_test[:, 1]

# Convert approximate degree errors to km
lat_km = np.abs(lat_error) * 111
lon_km = np.abs(lon_error) * 111

# Approximate total distance error
distance_km = np.sqrt(lat_km**2 + lon_km**2)

print("===== CYCLONE TRACK PREDICTION =====")
print("Test samples:", len(y_test))

print("\nAverage latitude error:",
      round(np.mean(np.abs(lat_error)), 4), "degrees")

print("Average longitude error:",
      round(np.mean(np.abs(lon_error)), 4), "degrees")

print("Average distance error:",
      round(np.mean(distance_km), 2), "km")

print("Median distance error:",
      round(np.median(distance_km), 2), "km")

print("Maximum distance error:",
      round(np.max(distance_km), 2), "km")

print("\n===== EXAMPLE PREDICTION =====")
print("Actual LAT, LON:",
      y_test[0])

print("Predicted LAT, LON:",
      predictions[0])

# Save predictions
results = np.column_stack((y_test, predictions, distance_km))

np.savetxt(
    "outputs/predictions.csv",
    results,
    delimiter=",",
    header="actual_lat,actual_lon,predicted_lat,predicted_lon,error_km",
    comments=""
)

print("\nPredictions saved:")
print("outputs/predictions.csv")