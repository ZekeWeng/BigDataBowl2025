
import torch
import numpy as np
import polars as pl
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np

class OffensiveDataset(Dataset):
    def __init__(self, plays, labels):
        self.plays = plays
        self.labels = labels

    def __len__(self):
        return len(self.plays)

    def __getitem__(self, idx):
        return self.plays[idx], self.labels[idx]


def create_tensors(df):
    def create_tensor_helper(df):
        values = df.select(METRICS).to_numpy()
        relative_matrices = values[:, None, :] - values[None, :, :]
        return torch.tensor(relative_matrices, dtype=torch.float32)
    plays = []
    for _, play_df in df.group_by(["gameId", "playId"], maintain_order=True):
        play_tensors = []
        for _, frame_df in play_df.group_by(["frameId"], maintain_order=True):
            frame_tensor = create_tensor_helper(frame_df)
            play_tensors.append(frame_tensor)
        play_tensor = torch.stack(play_tensors)
        plays.append(play_tensor)
    return plays

def collate_fn(batch):
    plays, labels = zip(*batch)
    batch_size = len(plays)
    max_frames = 960 #max([play.size(0) for play in plays]) #960
    max_players = max([play.size(1) for play in plays]) # 12

    padded_plays = torch.zeros(batch_size, max_frames, max_players, max_players, len(METRICS))
    masks = torch.zeros(batch_size, max_frames, dtype=torch.bool)

    for i, play in enumerate(plays):
        num_frames = play.size(0)
        num_players = play.size(1)
        padded_plays[i, :num_frames, :num_players, :num_players, :] = play
        masks[i, :num_frames] = 1

    labels = torch.tensor(labels)
    return padded_plays, labels, masks

if __name__ == "__main__":
    METRICS = ['x', 'y', 's', 'a', 'dis', 'o', 'dir', 'xs', 'ys']
    presnap_offense = pl.read_csv("data/outputs/presnap_offense.csv", null_values=["NA"])

    train = presnap_offense.filter(pl.col("week") <=7)
    val = presnap_offense.filter(pl.col("week") == 8)
    test = presnap_offense.filter(pl.col("week") == 9)

    train_tensors = create_tensors(train)
    val_tensors = create_tensors(val)
    test_tensors = create_tensors(test)

    train_labels = (
        train.group_by(["gameId", "playId"]).agg(pl.col("offensivePlay").first()
        .alias("first_offensivePlay")).select("first_offensivePlay").to_series().to_list()
    )

    val_labels = (
        val.group_by(["gameId", "playId"]).agg(pl.col("offensivePlay").first()
        .alias("first_offensivePlay")).select("first_offensivePlay").to_series().to_list()
    )

    test_labels = (
        test.group_by(["gameId", "playId"]).agg(pl.col("offensivePlay").first()
        .alias("first_offensivePlay")).select("first_offensivePlay").to_series().to_list()
    )

    train_dataset = OffensiveDataset(train_tensors, train_labels)
    val_dataset = OffensiveDataset(val_tensors, val_labels)
    test_dataset = OffensiveDataset(test_tensors, test_labels)

    torch.save({
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "params": {"batch_size": 32, "shuffle": False},
        "collate_fn": collate_fn
    }, "dataloaders.pth")