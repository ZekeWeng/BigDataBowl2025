import pandas as pd
import os
import CONSTANTS


def preprocess_plays(df):
    df['playId'] = df.groupby('gameId')['playId'].rank(method='dense').astype(int)
    df['Id'] = df['gameId'].astype(str) + '_' + df['playId'].astype(str)

    df['penaltyYards'].fillna(0).astype(int)
    df['qbSpike'] = df['qbSpike'].fillna(0).astype('bool')
    df['qbKneel'] = df['qbKneel'].astype(bool)
    return df.loc[~(df['qbSpike']) & ~(df['qbKneel'])].copy()


def preprocess_player_play(df):
    df['playId'] = df.groupby('gameId')['playId'].rank(method='dense').astype(int)
    df['Id'] = df['gameId'].astype(str) + '_' + df['playId'].astype(str)
    df.rename(columns={"penalty_yards": "player_penalty_yards"})
    return df.copy()


def preprocess_tracking_helper(tracking):
    tracking.loc[tracking['playDirection'] == 'left', 'x'] = 120 - tracking.loc[tracking['playDirection'] == 'left', 'x']
    tracking.loc[tracking['playDirection'] == 'left', 'y'] = (160/3) - tracking.loc[tracking['playDirection'] == 'left', 'y']
    tracking.loc[tracking['playDirection'] == 'left', 'dir'] += 180
    tracking.loc[tracking['dir'] > 360, 'dir'] -= 360
    tracking.loc[tracking['playDirection'] == 'left', 'o'] += 180
    tracking.loc[tracking['o'] > 360, 'o'] -= 360

    tracking['playId'] = tracking.groupby('gameId')['playId'].rank(method='dense').astype(int)
    tracking['Id'] = tracking['gameId'].astype(str) + '_' + tracking['playId'].astype(str)

    tracking.sort_values(['gameId', 'playId', 'frameId', 'club']).reset_index(drop=True)
    return tracking


def preprocess_presnap(df):
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
    df.sort_values(['gameId', 'playId', 'frameId', 'club', 'nflId']).reset_index(drop=True)

    return df


def preprocess(tracking_with_plays):
    tracking_with_plays['is_on_offense'] = tracking_with_plays['club'] == tracking_with_plays['possessionTeam']
    tracking_with_plays['is_on_defense'] = tracking_with_plays['club'] == tracking_with_plays['defensiveTeam']
    return tracking_with_plays


def preprocess_tracking(presnap=False):
    tracking = []
    for week in range(1, 10):
        TRACKING = pd.read_csv(f"data/2025/raw/tracking_week_{week}.csv")
        tracking_week = preprocess_tracking_helper(TRACKING)
        if presnap:
            tracking_week = preprocess_presnap(tracking_week)
        tracking.append(tracking_week)

    return pd.concat(tracking, axis=0, ignore_index=True)


if __name__ == '__main__':
    # Load Data
    GAMES = pd.read_csv(CONSTANTS.GAMES)
    PLAYERS = pd.read_csv(CONSTANTS.PLAYERS, usecols=CONSTANTS.PRESNAP_PLAYERS_COLS)
    PLAYS = pd.read_csv(CONSTANTS.PLAYS, usecols=CONSTANTS.PRESNAP_PLAYS_COLS)
    PLAYER_PLAY = pd.read_csv(CONSTANTS.PLAYER_PLAY, usecols=CONSTANTS.PRESNAP_PLAYER_PLAY_COLS)
    print("Finished Loading Data")

    # Preprocess
    plays = preprocess_plays(PLAYS)
    player_play = preprocess_player_play(PLAYER_PLAY)
    tracking = preprocess_tracking()
    print("Finished Preprocessing Data")

    # Merge Data
    plays_with_players = pd.merge(plays, player_play, how="left", on=['Id','gameId', 'playId'])
    tracking_with_plays = pd.merge(tracking, plays_with_players, on=['Id', 'nflId', 'gameId', 'playId'], how='left')
    df = preprocess(tracking_with_plays)
    df = pd.merge(df, GAMES, how="left", on=['gameId'])
    df = pd.merge(df, PLAYERS, how="left", on=['nflId', 'displayName'])
    print("Finished Merging Data")

    output_file = f'{CONSTANTS.OUTPUT_PATH}/processed_1-5.csv'
    df.to_csv(output_file, index=False)
    print("Finished Preprocesssing")

