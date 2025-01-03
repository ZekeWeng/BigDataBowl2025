import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import utils

mappings = {
    "pff_manZone": {
        "Zone": 0, "Man": 1
    },
    "defCoverage": {
        "Cover-0": 0, "Cover-1": 1, "Cover-2": 2, "Cover-3": 3, "Cover-6": 4,
        "Quarters": 5,
    },
    "runPass": {
        "RUN": 0, "PASS": 1
    }
}

class BigDataBowlDatasetFeatures(Dataset):
    def __init__(self, df: pl.DataFrame, fra_features, max_frames=256):
        self.max_frames = max_frames
        self.fra_features = fra_features
        self.df = df

        grouped = (
            df.group_by(["gameId", "playId"], maintain_order=True)
              .agg([
                  pl.count("gameId").alias("row_count"),
              ])
        )
        grouped = grouped.filter(pl.col("row_count") % 22 == 0)

        self.valid_groups = [tuple(row) for row in grouped.select(["gameId", "playId"]).to_numpy()]

    def __len__(self):
        return len(self.valid_groups)

    def __getitem__(self, idx):
        game_id, play_id = self.valid_groups[idx]

        play_df = self.df.filter(
            (pl.col("gameId") == game_id) & (pl.col("playId") == play_id)
        )

        frames = []
        for _, df_frame in play_df.group_by("frameId", maintain_order=True):
            offense_df = df_frame.filter(pl.col("club") == pl.col("possessionTeam"))
            defense_df = df_frame.filter(pl.col("club") == pl.col("defensiveTeam"))

            offense_np = offense_df.select(self.fra_features).to_numpy()        # [11, #features]
            defense_np = defense_df.select(self.fra_features).to_numpy()        # [11, #features]

            offense_expanded = offense_np[:, None, :].repeat(11, axis=1)        # [11, 11, #features]
            defense_expanded = defense_np[None, :, :].repeat(11, axis=0)        # [11, 11, #features]
            relative_np = offense_expanded - defense_expanded                   # [11, 11, #features]

            combined_np = np.concatenate([
                offense_expanded,
                defense_expanded,
                relative_np],
                axis=-1)                                                        # [11, 11, #features*3]
            combined_np = np.moveaxis(combined_np, -1, 0)                       # [#features*3, 11, 11]

            frames.append(torch.tensor(combined_np, dtype=torch.float32))



        f, p, p = frames[0].shape                                               # [#features, 11, 11]

        play_shape = (self.max_frames, f, p, p)
        play_tensor = torch.full(play_shape, -1e9, dtype=torch.float32)

        for i in range(min(len(frames), self.max_frames)):
            play_tensor[i] = frames[i]

        return play_tensor

class BigDataBowlDatasetLabels(Dataset):
    def __init__(self, df: pl.DataFrame, label_col: str, max_frames: int = 256):
        self.max_frames = max_frames
        grouped = (
            df.group_by(["gameId", "playId"], maintain_order=True)
              .agg([
                  pl.count("gameId").alias("row_count"),
                  pl.first(label_col).alias("label_val")
              ])
        )
        grouped = grouped.filter(pl.col("row_count") % 22 == 0)

        self.labels = grouped.select("label_val").to_numpy().flatten()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.full((self.max_frames,), self.labels[idx], dtype=torch.float32)

class BigDataBowlDatasetLengths(Dataset):
    def __init__(self, df: pl.DataFrame):
        self.df = df
        grouped = (
            df.group_by(["gameId", "playId"], maintain_order=True)
              .agg([
                  pl.count("gameId").alias("row_count"),
                  pl.col("frameId").max().alias("numFrames"),
              ])
        )

        grouped = grouped.filter(pl.col("row_count") % 22 == 0)

        self.frame_lengths = (grouped.select(["numFrames"]).to_numpy()).flatten()

    def __len__(self):
        return len(self.frame_lengths)

    def __getitem__(self, idx):
        return self.frame_lengths[idx]

def save_dataset(dataset: Dataset, file_path: str, batch_size=128):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_labs = []

    for labs in data_loader:
        all_labs.append(labs)

    all_labs = torch.cat(all_labs, dim=0)

    torch.save(all_labs, file_path)
    print(f"Saved label dataset to {file_path}, shape={all_labs.shape}")

if __name__ == "__main__":
    keys = ["week", "gameId", "playId", "frameId"]
    label_cols = ["pff_manZone", "defCoverage", "runPass"]
    frame_cols = ["relative_x", "relative_y", "xs", "ys", "xa", "ya", "dis", "o"]
    teams_cols = ["club", "possessionTeam", "defensiveTeam"]
    columns = keys + label_cols + frame_cols + teams_cols

    dfs = []
    for week in range(1, 10):
        df_week = pl.read_csv(
            f"{utils.presnap_week}/week_{week}.csv",
            null_values=["NA"],
            columns=columns
        )
        dfs.append(df_week)

    df = pl.concat(dfs).with_columns([
        pl.col(col_name)
          .replace(mappings[col_name])
          .alias(col_name)
          .cast(pl.Float32)
        for col_name in label_cols
    ])

    df_train = df.filter(pl.col("week") <= 7)
    df_val   = df.filter(pl.col("week") == 8)
    df_test  = df.filter(pl.col("week") == 9)

    batch_size = 128
    max_frames = 256

    ### Labels

    path = f"{utils.tensors}/Labels/"
    for name_col in label_cols:
        val_dataset = BigDataBowlDatasetLabels(df_val, label_col=name_col, max_frames=max_frames)
        save_dataset(val_dataset, f"{path}{name_col}_val.pt", batch_size=batch_size)

        test_dataset = BigDataBowlDatasetLabels(df_test, label_col=name_col, max_frames=max_frames)
        save_dataset(test_dataset, f"{path}{name_col}_test.pt", batch_size=batch_size)

        train_dataset = BigDataBowlDatasetLabels(df_train, label_col=name_col, max_frames=max_frames)
        save_dataset(train_dataset, f"{path}{name_col}_train.pt", batch_size=batch_size)

    ### Lengths

    path = f"{utils.tensors}/Model/Lengths/"
    val_dataset = BigDataBowlDatasetLengths(df_val)
    save_dataset(val_dataset, f"{path}lengths_val.pt", batch_size=batch_size)

    test_dataset = BigDataBowlDatasetLengths(df_test)
    save_dataset(test_dataset, f"{path}lengths_test.pt", batch_size=batch_size)

    train_dataset = BigDataBowlDatasetLengths(df_train)
    save_dataset(train_dataset, f"{path}lengths_train.pt", batch_size=batch_size)


    ### Features

    path = f"{utils.tensors}/Model/Features/"
    val_dataset = BigDataBowlDatasetFeatures(df_val, fra_features=frame_cols, max_frames=max_frames)
    save_dataset(val_dataset, f"{path}features_val.pt", batch_size=batch_size)

    test_dataset = BigDataBowlDatasetFeatures(df_test, fra_features=frame_cols, max_frames=max_frames)
    save_dataset(test_dataset, f"{path}features_test.pt", batch_size=batch_size)

    train_dataset = BigDataBowlDatasetFeatures(df_train, fra_features=frame_cols, max_frames=max_frames)
    save_dataset(train_dataset, f"{path}features_train.pt", batch_size=batch_size)
