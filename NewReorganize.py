import pandas as pd

# Load the CSV file
df = pd.read_csv('Entire_Input.csv')

# Initialize a list to hold the reordered rows
reordered_rows = []

# Determine the total number of rows in the DataFrame
total_rows = len(df)
print(total_rows)

# The size of each block to be reordered
block_size = 4500

# Calculate the number of blocks
num_blocks = total_rows // block_size

# Iterate through each block
for block in range(num_blocks):
    # Calculate the starting row index for the current block
    start_row = block * block_size
    # Iterate through each row index for the reordering pattern within the current block
    for i in range(900):  # Loop through the first 900 rows of the block
        for j in range(0, block_size, 900):  # Iterate over each set of 900 rows within the block
            # Calculate the actual row index within the entire DataFrame
            row_index = start_row + i + j
            # Append the row as a list to 'reordered_rows'
            reordered_rows.append(df.iloc[row_index].tolist())

# Create a new DataFrame from the reordered rows
new_df = pd.DataFrame(reordered_rows, columns=df.columns)

# Save the new DataFrame to a CSV file
new_df.to_csv('New_Entire_Input.csv', index=False)
