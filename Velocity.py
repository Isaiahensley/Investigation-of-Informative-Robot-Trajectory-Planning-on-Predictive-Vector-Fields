import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_pred = 'predictions.csv'
csv_actual = 'Testing_Target.csv'
df_pred = pd.read_csv(csv_pred)
df_actual = pd.read_csv(csv_actual)

# Extract actual and predicted values
actual_u = df_actual['u']
actual_v = df_actual['v']
predicted_u = df_pred['u']
predicted_v = df_pred['v']
lat = df_actual['lat']  # Assuming 'lat' column contains latitude values
lon = df_actual['lon']  # Assuming 'lon' column contains longitude values

# Plotting quiver plot for actual values
fig, ax = plt.subplots(1, 2, figsize=(18, 8))  # Create subplots

# Adjust the scale parameter for shorter arrows
scale = 8  # Experiment with different values to get the desired arrow length

# Quiver plot for actual values
ax[0].quiver(lon, lat, actual_u, actual_v, color='blue', label='Actual', angles='xy', scale_units='xy', scale=scale)
ax[0].set_xlabel('Longitude')
ax[0].set_ylabel('Latitude')
ax[0].set_title('Quiver Plot for Actual u, v values (First 900 rows)')
ax[0].legend()

# Quiver plot for predicted values
ax[1].quiver(lon, lat, predicted_u, predicted_v, color='red', label='Predicted', angles='xy', scale_units='xy', scale=scale)
ax[1].set_xlabel('Longitude')
ax[1].set_ylabel('Latitude')
ax[1].set_title('Quiver Plot for Predicted u, v values (First 900 rows)')
ax[1].legend()

plt.show()
