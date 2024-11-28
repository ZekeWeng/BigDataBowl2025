import pandas as pd
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from visualization import animate_play
import torch
from torch_geometric.data import Data, DataLoader

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder


# def encode_features(frame):
#     features = [;pso]
#     encoder = OneHotEncoder(sparse=False)
#     categorical_features = frame[features].to_numpy()  # Replace 'team' with your categorical feature(s)
#     encoded_features = encoder.fit_transform(categorical_features)
#     return encoded_features

def create_edges(positions):
    num_nodes = positions.shape[0]
    edge_index = []
    edge_attr = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edge_index.append([i, j])
            edge_index.append([j, i])

            distance = np.linalg.norm(positions[i] - positions[j])
            edge_attr.extend([distance, distance])
    return edge_index, edge_attr

def process_play(df):
    frames = []

    for _, frame in df.groupby('frameId'):
        positions = frame[['x', 'y']].to_numpy()
        node_continuous_features = frame[['x', 'y', 'o', 's', 'a']].to_numpy()

        # node_encoded_features = encode_features(frame)
        # node_attributes = np.concatenate([node_continuous_features, node_encoded_features], axis=1)
        node_attributes = node_continuous_features
        node_features = torch.tensor(node_attributes, dtype=torch.float)

        edge_index, edge_attributes = create_edges(positions)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(edge_attributes, dtype=torch.float)

        frames.append(Data(x=node_features, edge_index=edge_index, edge_attr=edge_features))
    return frames

def process_all_plays(df):
    return df.groupby('Id').apply(lambda play: process_play(play)).tolist()

def create_kfold_splits(df, target_values, k=5, random_state=42):
    kfolds = KFold(n_splits=k, shuffle=True, random_state=random_state)
    target_values = np.array(target_values)

    folds = []

    for train_indices, test_indices in kfolds.split(df):
        train_X = [df[i] for i in train_indices]
        train_y = target_values[train_indices]

        test_X = [df[i] for i in test_indices]
        test_y = target_values[test_indices]

        folds.append((train_X, train_y, test_X, test_y))
    return folds

def create_kfold_splits(df, target_values, k=5, random_state=42):
    kfolds = KFold(n_splits=k, shuffle=True, random_state=random_state)
    plays = process_all_plays(df)
    target_values = np.array(target_values)

    folds = []

    for train_indices, test_indices in kfolds.split(plays):
        train_X = [plays[i] for i in train_indices]
        train_y = target_values[train_indices]

        test_X = [plays[i] for i in test_indices]
        test_y = target_values[test_indices]

        folds.append((train_X, train_y, test_X, test_y))
    return folds

if __name__ == "__main__":
    presnap = pd.read_csv("data/2025/processed/data.csv")

    data = presnap.query("displayName != 'football'")
    mean_yards_gained = data.groupby("Id")['yardsGained'].transform("mean")
    df = data[~mean_yards_gained.isnull()]

    plays = process_all_plays(df)
    targets = df.groupby("Id")['yardsGained'].mean().values

    create_kfold_splits(plays, targets)