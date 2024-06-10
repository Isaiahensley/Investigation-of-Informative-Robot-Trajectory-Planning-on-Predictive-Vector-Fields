import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
csv_pred = 'predictions.csv'
csv_actual = 'Testing_Target.csv'
df_pred = pd.read_csv(csv_pred)
df_actual = pd.read_csv(csv_actual)

# Use only the first 900 rows
#df_results_subset = df_results.head(900)

# Extract actual and predicted temperature values
actual_temp = df_actual['temp']
predicted_temp = df_pred['temp']
lat = df_actual['lat']  # Assuming 'lat' column contains latitude values
lon = df_actual['lon']  # Assuming 'lon' column contains longitude values

# Reshape the temperature values into a 2D grid
actual_temp_grid = actual_temp.values.reshape(len(set(lat)), -1)
predicted_temp_grid = predicted_temp.values.reshape(len(set(lat)), -1)

# Plotting heatmaps for actual and predicted temperature values
fig, ax = plt.subplots(1, 2, figsize=(18, 8))  # Create subplots

# Heatmap for actual temperature values
sns.heatmap(actual_temp_grid, cmap='coolwarm', cbar_kws={'label': 'Temperature'}, ax=ax[0])
ax[0].set_xlabel('Longitude')
ax[0].set_ylabel('Latitude')
ax[0].set_title('Heatmap of Actual Temperature (First 900 rows)')

# Heatmap for predicted temperature values
sns.heatmap(predicted_temp_grid, cmap='coolwarm', cbar_kws={'label': 'Temperature'}, ax=ax[1])
ax[1].set_xlabel('Longitude')
ax[1].set_ylabel('Latitude')
ax[1].set_title('Heatmap of Predicted Temperature (First 900 rows)')

plt.show()
