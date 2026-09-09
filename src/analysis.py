import numpy as np
import tensorflow as tf

print("Loading test data...")

X_test = np.load("Data/processed/X_test.npy")
y_test = np.load("Data/processed/y_test.npy")

print("Loading model...")

model = tf.keras.models.load_model("models/cyclone_lstm.keras")

print("Making predictions...")

predictions = model.predict(X_test, verbose=0)

print("Predictions completed!")

# Errors
lat_error = predictions[:, 0] - y_test[:, 0]
lon_error = predictions[:, 1] - y_test[:, 1]

# Approximate conversion: 1 degree latitude/longitude ≈ 111 km
lat_km = np.abs(lat_error) * 111
lon_km = np.abs(lon_error) * 111

distance_km = np.sqrt(lat_km**2 + lon_km**2)

print()
print("===== CYCLONE TRACK PREDICTION =====")
print("Test samples:", len(y_test))

print("Average latitude error:",
      round(np.mean(np.abs(lat_error)), 4), "degrees")

print("Average longitude error:",
      round(np.mean(np.abs(lon_error)), 4), "degrees")

print("Average distance error:",
      round(np.mean(distance_km), 2), "km")

print("Median distance error:",
      round(np.median(distance_km), 2), "km")

print("Maximum distance error:",
      round(np.max(distance_km), 2), "km")

print()
print("===== EXAMPLE PREDICTION =====")

print("Actual LAT, LON:")
print(y_test[0])

print("Predicted LAT, LON:")
print(predictions[0])

# Save results
results = np.column_stack(
    (y_test, predictions, distance_km)
)

np.savetxt(
    "outputs/predictions.csv",
    results,
    delimiter=",",
    header="actual_lat,actual_lon,predicted_lat,predicted_lon,error_km",
    comments=""
)

print()
print("Predictions saved:")
print("outputs/predictions.csv")