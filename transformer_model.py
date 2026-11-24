#  Transformer.py
# -----------------------------------------------------------------------------
#  Transformer Model for Seismic Response Prediction of Bridge Piers
#
#  Authors: Jianwei Zhang, Chao Yu, Xiya Liu
#  Affiliation: College of Civil Engineering, Tongji University
#
#  Copyright (c) 2025 Jianwei Zhang, Chao Yu, Xiya Liu. All rights reserved.
# -----------------------------------------------------------------------------

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy import signal
from scipy.fft import fft, fftfreq

# Check whether a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Define the dataset class
class BridgeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Feature extraction function
def extract_acc_features(acc_data, dt=0.02):
    """
    Extract features from a ground motion acceleration time history.
    """
    acc = np.array(acc_data)

    # Basic statistical features
    mean_acc = np.mean(acc)
    std_acc = np.std(acc)
    max_acc = np.max(acc)
    min_acc = np.min(acc)
    abs_max_acc = np.max(np.abs(acc))
    rms_acc = np.sqrt(np.mean(acc**2))

    # Frequency-domain features
    N = len(acc)
    freqs = fftfreq(N, dt)
    fft_vals = fft(acc)
    magnitude = np.abs(fft_vals)

    # Dominant frequency and spectral features
    main_freq_idx = np.argmax(magnitude[1:N//2]) + 1  # Skip the DC component
    main_freq = freqs[main_freq_idx]
    dominant_freq_energy = magnitude[main_freq_idx] / np.sum(magnitude[1:N//2])

    # Energy features
    total_energy = np.sum(acc**2)
    energy_25 = np.sum(acc[:N//4]**2)
    energy_50 = np.sum(acc[N//4:N//2]**2)
    energy_75 = np.sum(acc[N//2:3*N//4]**2)
    energy_100 = np.sum(acc[3*N//4:]**2)

    # Peak features
    peaks, _ = signal.find_peaks(acc, height=0.1*np.std(acc))
    num_peaks = len(peaks)
    if len(peaks) > 0:
        avg_peak_height = np.mean(acc[peaks])
        avg_peak_distance = np.mean(np.diff(peaks)) if len(peaks) > 1 else 0
    else:
        avg_peak_height = 0
        avg_peak_distance = 0

    # Duration feature
    significant_duration = np.sum(np.abs(acc) > 0.05 * abs_max_acc) * dt

    # Return the extracted features
    features = [
        mean_acc, std_acc, max_acc, min_acc, abs_max_acc, rms_acc,
        main_freq, dominant_freq_energy, total_energy,
        energy_25, energy_50, energy_75, energy_100,
        num_peaks, avg_peak_height, avg_peak_distance, significant_duration
    ]

    return features


# Define the enhanced Transformer model
class EnhancedTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(EnhancedTransformerModel, self).__init__()

        # Feature extraction layer - used to process the ground motion time history
        self.acc_feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 25, 128),  # Assuming an original length of 100, reduced to 25 after two pooling steps
            nn.ReLU()
        )

        # Bridge parameter feature extractor
        self.bridge_feature_extractor = nn.Sequential(
            nn.Linear(4, 64),  # 4 bridge parameters
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # Transformer after feature combination
        self.feature_combiner = nn.Linear(128 + 128, hidden_dim)  # Bridge features + ground motion features

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Additional feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim // 4, output_dim)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.size(0)

        # Separate bridge parameters and ground motion data
        bridge_params = x[:, :4]   # First 4 values are bridge parameters
        acc_data = x[:, 4:104]     # Next 100 values are ground motion data

        # Extract bridge parameter features
        bridge_features = self.bridge_feature_extractor(bridge_params)

        # Reshape ground motion data for CNN processing
        acc_cnn_input = acc_data.unsqueeze(1)  # (batch_size, 1, 100)
        acc_features = self.acc_feature_extractor(acc_cnn_input)

        # Combine features
        combined_features = torch.cat((bridge_features, acc_features), dim=1)
        x = self.feature_combiner(combined_features)

        # Expand into sequence format
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # Add positional encoding
        pos_encoding = self.pos_encoding.expand(batch_size, -1, -1)
        x = x + pos_encoding

        # Transformer processing
        x = self.transformer_encoder(x)

        # Process the output
        x = self.feature_processor(x[:, -1, :])  # Take the output of the last position
        output = self.output_layer(x)

        return output


# Load data
def load_data(results_dir, acc_dir):
    """
    Load data from the results directory and the ground motion directory.
    """
    all_data = []

    # Get all sample folders
    sample_dirs = [d for d in Path(results_dir).iterdir() if d.is_dir() and d.name.startswith('sample_')]

    print(f"Found {len(sample_dirs)} sample folders")

    for i, sample_dir in enumerate(sample_dirs):
        print(f"Processing sample folder: {sample_dir.name}")

        # Read the CSV result file
        csv_files = list(sample_dir.glob('sample_*_results.csv'))
        if not csv_files:
            print(f"  Skipping {sample_dir.name} - no CSV file found")
            continue

        csv_path = csv_files[0]
        print(f"  Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)

        # For each ground motion record, read the corresponding acceleration file
        for _, row in df.iterrows():
            gm_id = int(row['gm_id'])
            max_disp = row['max_disp_mid_m']
            span_length = row['span_length_m']
            pier_height = row['pier_height_m']
            pier_diameter = row['pier_diameter_m']
            deck_mass = row['deck_mass_t_per_m']

            # Read the acceleration file from the ground motion directory
            acc_file = Path(acc_dir) / f"{gm_id}.acc"

            if not acc_file.exists():
                print(f"    Skipping ground motion record {gm_id} - acceleration file not found: {acc_file}")
                continue

            print(f"    Reading acceleration file: {acc_file}")

            # Read the ground motion acceleration time history
            try:
                with open(acc_file, 'r') as f:
                    lines = f.readlines()

                acc_data = []
                for line in lines:
                    values = line.strip().split()
                    for val in values:
                        try:
                            acc_data.append(float(val))
                        except ValueError:
                            continue
            except Exception as e:
                print(f"    Failed to read acceleration file: {e}")
                continue

            # Limit the acceleration data length (take the first 100 points)
            max_len = 100
            if len(acc_data) > max_len:
                acc_data = acc_data[:max_len]
            else:
                acc_data += [0.0] * (max_len - len(acc_data))  # Pad to a fixed length

            # Extract ground motion features
            acc_features = extract_acc_features(acc_data)

            # Build the feature vector: [span_length, pier_height, pier_diameter, deck_mass] + ground motion features + ground motion time history
            features = [span_length, pier_height, pier_diameter, deck_mass] + acc_features + acc_data
            target = [max_disp]

            all_data.append((features, target))

    if len(all_data) == 0:
        raise ValueError("No valid data files were found")

    # Convert to numpy arrays
    X = np.array([item[0] for item in all_data])
    y = np.array([item[1] for item in all_data])

    return X, y


# Main function
def main():
    # Resolve the script directory so all default paths are relative to the repo
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Enhanced Transformer surrogate model for bridge pier seismic response prediction."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=base_dir / "time_history_results",
        help="Directory containing the per-sample result folders (default: ./time_history_results)",
    )
    parser.add_argument(
        "--acc-dir",
        type=Path,
        default=base_dir / "waves",
        help="Directory containing the .acc ground motion files (default: ./waves)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    acc_dir = args.acc_dir

    # Check whether the data directories exist
    if not results_dir.exists():
        print(f"Results directory does not exist: {results_dir}")
        return

    if not acc_dir.exists():
        print(f"Ground motion directory does not exist: {acc_dir}")
        return

    print("Loading data...")
    try:
        X, y = load_data(results_dir, acc_dir)
        print(f"Loaded {X.shape[0]} samples, feature dimension: {X.shape[1]}")
    except Exception as e:
        print(f"Error while loading data: {e}")
        return

    # Split into training, validation, and test sets (60%, 20%, 20%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Standardize the features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)

    # Create data loaders
    train_dataset = BridgeDataset(X_train_scaled, y_train_scaled)
    val_dataset = BridgeDataset(X_val_scaled, y_val_scaled)
    test_dataset = BridgeDataset(X_test_scaled, y_test_scaled)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Reduce the batch size
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Define model parameters
    input_dim = X.shape[1]  # Input feature dimension
    hidden_dim = 256
    num_heads = 8
    num_layers = 6
    output_dim = 1  # Output displacement response

    # Create the model
    model = EnhancedTransformerModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        output_dim=output_dim
    ).to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)

    # Train the model
    num_epochs = 500
    train_losses = []
    val_losses = []

    print("Start training the model...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if epoch % 20 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    # Plot the loss curves
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Evaluate the model on the test set
    model.eval()
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            test_predictions.extend(outputs.cpu().numpy())
            test_targets.extend(batch_y.cpu().numpy())

    # Inverse-transform the predictions and true values
    test_predictions = scaler_y.inverse_transform(np.array(test_predictions))
    test_targets = scaler_y.inverse_transform(np.array(test_targets))

    # Compute the evaluation metrics
    mse = mean_squared_error(test_targets, test_predictions)
    mae = mean_absolute_error(test_targets, test_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_targets, test_predictions)

    print("\nTest set evaluation metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2: {r2:.6f}")

    # Plot predicted values vs. true values
    plt.subplot(1, 3, 2)
    plt.scatter(test_targets, test_predictions, alpha=0.6)
    plt.plot([test_targets.min(), test_targets.max()], [test_targets.min(), test_targets.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs True Values\nR2 = {r2:.4f}')
    plt.grid(True)

    # Plot the residual plot
    plt.subplot(1, 3, 3)
    residuals = test_targets.flatten() - test_predictions.flatten()
    plt.scatter(test_predictions, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print some prediction examples
    print("\nPrediction examples (first 10):")
    for i in range(min(10, len(test_targets))):
        print(f"True value: {test_targets[i][0]:.6f}, Predicted value: {test_predictions[i][0]:.6f}")


if __name__ == "__main__":
    main()
