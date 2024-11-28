import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
offense = pl.read_csv("data/outputs/presnap_offense.csv")
O = offense.filter(pl.col("week") == 1)

def create_frame_tensor_optimized(df, metrics):
    values = df.select(metrics).to_numpy()
    relative_matrices = values[:, None, :] - values[None, :, :]
    return torch.tensor(relative_matrices, dtype=torch.float32)

def process_plays_optimized(df, metrics):
    plays = []
    for play_key, play_df in df.group_by(["gameId", "playId"], maintain_order=True):
        play_tensors = []
        for _, frame_df in play_df.group_by(["frameId"], maintain_order=True):
            frame_tensor = create_frame_tensor_optimized(frame_df, metrics)
            play_tensors.append(frame_tensor)
        play_tensor = torch.stack(play_tensors)
        plays.append(play_tensor)
    return plays

metrics = ['x', 'y', 's', 'a', 'dis', 'o', 'dir', 'xs', 'ys']
plays = process_plays_optimized(O, metrics)

# Extract labels
labels = O.group_by(["gameId", "playId"]).agg(pl.col("isDropback").first())["isDropback"].to_list()

class FootballPlayDataset(Dataset):
    def __init__(self, plays, labels):
        self.plays = plays
        self.labels = labels

    def __len__(self):
        return len(self.plays)

    def __getitem__(self, idx):
        return self.plays[idx], self.labels[idx]

dataset = FootballPlayDataset(plays, labels)

def collate_fn(batch):
    plays, labels = zip(*batch)
    batch_size = len(plays)
    max_frames = max([play.size(0) for play in plays])
    max_players = max([play.size(1) for play in plays])

    # Initialize tensors for padded plays and masks
    padded_plays = torch.zeros(batch_size, max_frames, max_players, max_players, len(metrics))
    masks = torch.zeros(batch_size, max_frames, dtype=torch.bool)

    for i, play in enumerate(plays):
        num_frames = play.size(0)
        num_players = play.size(1)
        padded_plays[i, :num_frames, :num_players, :num_players, :] = play
        masks[i, :num_frames] = 1

    labels = torch.tensor(labels)
    return padded_plays, labels, masks

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# Frame-level CNN
class FrameCNN(nn.Module):
    def __init__(self, input_channels, embedding_dim):
        super(FrameCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),  # Output: (32, N, N)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),              # Output: (64, N, N)
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, kernel_size=3, padding=1),   # Output: (embedding_dim, N, N)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Output: (embedding_dim, 1, 1)
        )

    def forward(self, x):
        x = self.cnn(x)                      # (batch_size*num_frames, embedding_dim, 1, 1)
        x = x.view(x.size(0), -1)            # (batch_size*num_frames, embedding_dim)
        return x

# Temporal Transformer Encoder
class TemporalTransformer(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, dropout=0.1):
        super(TemporalTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask):
        # x: (batch_size, seq_len, embedding_dim)
        x = x.transpose(0, 1)  # (seq_len, batch_size, embedding_dim)
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        output = output.transpose(0, 1)  # (batch_size, seq_len, embedding_dim)
        return output

# Combined Model
class RPClassifier(nn.Module):
    def __init__(self, input_channels, embedding_dim, num_heads, num_layers, num_classes):
        super(RPClassifier, self).__init__()
        self.frame_cnn = FrameCNN(input_channels, embedding_dim)
        self.temporal_transformer = TemporalTransformer(embedding_dim, num_heads, num_layers)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, masks):
        # x: (batch_size, max_seq_len, N, N, C)
        # masks: (batch_size, max_seq_len)
        batch_size, seq_len, N1, N2, C = x.size()
        x = x.view(batch_size * seq_len, N1, N2, C)
        x = x.permute(0, 3, 1, 2)  # (batch_size*seq_len, C, N1, N2)
        frame_embeddings = self.frame_cnn(x)  # (batch_size * seq_len, embedding_dim)
        frame_embeddings = frame_embeddings.view(batch_size, seq_len, -1)  # (batch_size, seq_len, embedding_dim)
        src_key_padding_mask = ~masks  # Invert mask for transformer (True indicates padding positions)
        transformer_output = self.temporal_transformer(frame_embeddings, src_key_padding_mask)
        # Use mean pooling over valid frames
        masks = masks.unsqueeze(-1)  # (batch_size, seq_len, 1)
        masked_output = transformer_output * masks  # Zero out padding positions
        sum_embeddings = masked_output.sum(dim=1)  # Sum over seq_len
        valid_lengths = masks.sum(dim=1)           # (batch_size, 1)
        play_embeddings = sum_embeddings / valid_lengths  # (batch_size, embedding_dim)
        logits = self.fc(play_embeddings)          # (batch_size, num_classes)
        return logits

# Model parameters
input_channels = len(metrics)
embedding_dim = 128
num_heads = 4
num_layers = 2
num_classes = len(set(labels))  # Number of unique labels in the dataset

# Initialize model, loss function, and optimizer
model = RPClassifier(input_channels, embedding_dim, num_heads, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y, masks in dataloader:
            x = x.to(device)
            y = y.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(x, masks)
            loss = criterion(outputs, y.long())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Train the model
train_model(model, dataloader, criterion, optimizer, num_epochs=10)