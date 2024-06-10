import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib  # For saving the scaler
import math

# Define the Transformer Model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_transformer_blocks, dropout_rate=0.1):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, embed_dim)  # Embed input features to a higher dimension
        self.pos_encoder = PositionalEncoding(embed_dim)  # Initialize the positional encoding
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
                                                        dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_transformer_blocks)
        self.output_layer = nn.Linear(embed_dim, 4)  # Assuming you want to predict 4 features: 'u', 'v', 'salt', 'temp'

    def forward(self, src):
        src = self.input_embedding(src)  # Embed input features
        src = src.permute(1, 0, 2)  # Transformer expects (seq_len, batch, features)
        src = self.pos_encoder(src)  # Apply positional encoding
        transformer_output = self.transformer_encoder(src)
        output = self.output_layer(transformer_output[-1])
        return output



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


# Load and preprocess data function
def load_and_preprocess_data(input_path, labels_path):
    df_input = pd.read_csv(input_path)
    df_labels = pd.read_csv(labels_path)

    # Include month, day, and hour for inputs
    input_features = ['month', 'day', 'hour', 'lat', 'lon', 'u', 'v', 'salt', 'temp']
    label_features = ['u', 'v', 'salt', 'temp']

    df_input = df_input[input_features]
    df_labels = df_labels[label_features]

    # Normalize input data
    input_scaler = StandardScaler()
    scaled_inputs = input_scaler.fit_transform(df_input)

    # Normalize label data separately
    label_scaler = StandardScaler()
    scaled_labels = label_scaler.fit_transform(df_labels)

    input_data = np.array(scaled_inputs).reshape(-1, 5, 9)  # Adjust for new input shape
    labels = np.array(scaled_labels)

    return input_data, labels, input_scaler, label_scaler




# Load data and scaler
input_data, labels, input_scaler, label_scaler = load_and_preprocess_data('New_Entire_Input.csv', 'Entire_Target.csv')

# Save both scalers
joblib.dump(input_scaler, 'input_scaler.pkl')
joblib.dump(label_scaler, 'label_scaler.pkl')

# Data preparation: Manually split the last 20% of the data for validation
total_samples = len(input_data)
split_index = int(total_samples * 0.8)  # Calculate the index at which to split

X_train = input_data[:split_index]
y_train = labels[:split_index]
X_val = input_data[split_index:]
y_val = labels[split_index:]

# Convert to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float))

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # Shuffling is not needed for time series data
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Model initialization
model = TimeSeriesTransformer(input_dim=9, embed_dim=8, num_heads=8, ff_dim=128, num_transformer_blocks=2)

# Training setup
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 20  # Example epoch count

# Training loop
for epoch in range(epochs):
    model.train()
    for batch, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)
        loss.backward()
        optimizer.step()

    # Validation step
    model.eval()
    with torch.no_grad():
        val_loss = sum(loss_fn(model(X_val_batch), y_val_batch) for X_val_batch, y_val_batch in val_loader) / len(
            val_loader)
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'time_series_transformer_model.pth')


# Function to make predictions on new data and save to CSV
def predict_and_save(model_path, test_input_path, input_scaler_path, label_scaler_path, output_path):
    model = TimeSeriesTransformer(input_dim=9, embed_dim=8, num_heads=4, ff_dim=128, num_transformer_blocks=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load both scalers
    input_scaler = joblib.load(input_scaler_path)
    label_scaler = joblib.load(label_scaler_path)

    new_data = pd.read_csv(test_input_path)
    input_features = ['month', 'day', 'hour', 'lat', 'lon', 'u', 'v', 'salt', 'temp']
    new_data = new_data[input_features]
    scaled_new_data = input_scaler.transform(new_data)
    input_data = np.array(scaled_new_data).reshape(-1, 5, 9)
    input_data = torch.tensor(input_data, dtype=torch.float)

    predictions = []
    with torch.no_grad():
        for i in range(len(input_data)):
            prediction = model(input_data[i].unsqueeze(0))
            predictions.append(prediction.numpy())

    predictions = np.array(predictions).reshape(-1, 4)
    predicted_data = label_scaler.inverse_transform(predictions)
    df_predictions = pd.DataFrame(predicted_data, columns=['u', 'v', 'salt', 'temp'])
    df_predictions.to_csv(output_path, index=False)
    print("Predictions saved to", output_path)


predict_and_save('time_series_transformer_model.pth', 'New_Testing_Input.csv', 'input_scaler.pkl', 'label_scaler.pkl', 'predictions.csv')
