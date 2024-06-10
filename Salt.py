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
actual_salt = df_actual['salt']
predicted_salt = df_pred['salt']
lat = df_actual['lat']  # Assuming 'lat' column contains latitude values
lon = df_actual['lon']  # Assuming 'lon' column contains longitude values

# Reshape the salt values into a 2D grid
actual_salt_grid = actual_salt.values.reshape(len(set(lat)), -1)
predicted_salt_grid = predicted_salt.values.reshape(len(set(lat)), -1)

# Plotting heatmaps for actual and predicted salt values
fig, ax = plt.subplots(1, 2, figsize=(18, 8))  # Create subplots

# Heatmap for actual salt values
sns.heatmap(actual_salt_grid, cmap='coolwarm', cbar_kws={'label': 'Salt'}, ax=ax[0])
ax[0].set_xlabel('Longitude')
ax[0].set_ylabel('Latitude')
ax[0].set_title('Heatmap of Actual Salt (First 900 rows)')

# Heatmap for predicted salt values
sns.heatmap(predicted_salt_grid, cmap='coolwarm', cbar_kws={'label': 'Salt'}, ax=ax[1])
ax[1].set_xlabel('Longitude')
ax[1].set_ylabel('Latitude')
ax[1].set_title('Heatmap of Predicted Salt (First 900 rows)')

plt.show()
