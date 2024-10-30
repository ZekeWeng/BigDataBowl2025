import pandas as pd
import os
import CONSTANTS


def create_directories():
    directories = [
        CONSTANTS.INPUT_PATH,
        os.path.join(CONSTANTS.OUTPUT_PATH, "presnap"),
        os.path.join(CONSTANTS.OUTPUT_PATH, "matchup")
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def preprocess(df):
    df['playId'] = df.groupby('gameId')['playId'].rank(method='dense').astype(int)
    df['Id'] = df['gameId'].astype(str) + '_' + df['playId'].astype(str)
    df['penaltyYards'] = df['penaltyYards'].fillna(0).astype(int)
    return df.sort_values(['gameId', 'playId']).reset_index(drop=True)


def preprocess_tracking(df):
    df['playId'] = df.groupby('gameId')['playId'].rank(method='dense').astype(int)
    df['Id'] = df['gameId'].astype(str) + '_' + df['playId'].astype(str)

    line_set_frames = (
        df[df['event'] == 'line_set'].groupby(['gameId', 'playId'])['frameId']
        .min().reset_index()
        .rename(columns={'frameId': 'frameId_line_set'})
    )

    snap_frames = (
        df[df['frameType'] == 'SNAP'].groupby(['gameId', 'playId'])['frameId']
        .min().reset_index()
        .rename(columns={'frameId': 'frameId_snap'})
    )

    # Identify 'line_set' and 'SNAP'
    valid_frames = pd.merge(
        line_set_frames, snap_frames,
        on=['gameId', 'playId'],
        how='inner'
    )

    valid_frames = valid_frames[valid_frames['frameId_line_set'] <= valid_frames['frameId_snap']]

    df = pd.merge(
        df, valid_frames,
        on=['gameId', 'playId'],
        how='inner'
    )

    # Filter for presnap + postlineset
    df = df[
        (df['frameId'] >= df['frameId_line_set']) &
        (df['frameId'] <= df['frameId_snap'])
    ].copy()

    df.drop(['frameId_line_set', 'frameId_snap'], axis=1, inplace=True)
    return df.sort_values(['gameId', 'playId', 'frameId', 'club', 'nflId']).reset_index(drop=True)


if __name__ == '__main__':
    GAMES = pd.read_csv(os.path.join(CONSTANTS.INPUT_PATH, "games.csv"))
    PLAYERS = pd.read_csv(os.path.join(CONSTANTS.INPUT_PATH, "players.csv"))
    PLAYS = pd.read_csv(os.path.join(CONSTANTS.INPUT_PATH, "plays.csv"))
    PLAYER_PLAY = pd.read_csv(os.path.join(CONSTANTS.INPUT_PATH, "player_play.csv"))

    tracking = []
    print(f"Processing plays...")
    plays = preprocess(PLAYS)
    print(f"Processing player_play...")
    player_play = preprocess(PLAYER_PLAY)

    for week in range(1, 10):
        print(f"Processing week {week} tracking...")
        TRACKING = pd.read_csv(os.path.join(CONSTANTS.INPUT_PATH, f"tracking_week_{week}.csv"))
        tracking.append(preprocess_tracking(TRACKING))
    tracking = pd.concat(tracking, axis=0, ignore_index=True)

    print(f"Merging data...")
    df = pd.merge(plays, player_play, how="left", on=['Id','gameId', 'playId'])
    df = pd.merge(tracking, df, how="inner", on=['Id', 'nflId', 'gameId', 'playId'])
    df = pd.merge(df, GAMES, how="left", on=['gameId'])
    df = pd.merge(df, PLAYERS, how="left", on=['nflId', 'displayName'])

    output_file = f'{CONSTANTS.OUTPUT_PATH}/data.csv'
    df.to_csv(output_file, index=False)
    print(f"Completed Preprocesssing")