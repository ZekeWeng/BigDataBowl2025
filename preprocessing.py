# PREPROCESSING
import polars as pl
import math

def preprocess_games(df):
    return df.drop(['gameDate', 'gameTimeEastern'])

def preprocess_players(df):
    return df.drop(['birthDate', 'collegeName', 'displayName'])

def preprocess_plays(df):
    df = df.with_columns(
        pl.col('playId')
        .rank(method='dense')
        .over('gameId')
        .cast(pl.Int32)
        .alias('playId')
    )

    columns_to_drop = [
        'yardlineSide', 'yardlineNumber',
        'preSnapHomeScore', 'preSnapVisitorScore',
        'playNullifiedByPenalty',
        'preSnapHomeTeamWinProbability', 'preSnapVisitorTeamWinProbability',
        'dropbackDistance', 'timeToThrow', 'timeInTackleBox', 'timeToSack',
        'passTippedAtLine', 'unblockedPressure',
        'penaltyYards', 'prePenaltyYardsGained', 'qbSpike',
        'qbKneel', 'qbSneak'
    ]
    df = df.filter(pl.col("qbKneel") != True)
    df = df.drop(columns_to_drop)
    df = df.filter(~pl.col("pff_passCoverage").is_null())
    return df.sort(['gameId', 'playId'])

def preprocess_player_play(df):
    df = df.with_columns(
        pl.col('playId')
        .rank(method='dense')
        .over('gameId')
        .cast(pl.Int32)
        .alias('playId')
    )

    columns_to_keep = [
        "gameId", "playId", "nflId", "rushingYards", "passingYards", "receivingYards",
        "inMotionAtBallSnap", "shiftSinceLineset", "motionSinceLineset",
        "wasRunningRoute", "routeRan",
        "pff_defensiveCoverageAssignment",
        "pff_primaryDefensiveCoverageMatchupNflId",
        "pff_secondaryDefensiveCoverageMatchupNflId"
    ]
    df = df.select(columns_to_keep)
    return df.sort(['gameId', 'playId'])

def preprocessing_tracking(df):
    df = df.with_columns(
        pl.col('playId')
        .rank(method='dense')
        .over('gameId')
        .cast(pl.Int32)
        .alias('playId')
    )
    df = df.with_columns([
        pl.when(pl.col('playDirection') == 'left')
        .then(120 - pl.col('x'))
        .otherwise(pl.col('x'))
        .alias('x'),
        pl.when(pl.col('playDirection') == 'left')
        .then((160 / 3) - pl.col('y'))
        .otherwise(pl.col('y'))
        .alias('y'),
        pl.when(pl.col('playDirection') == 'left')
        .then((pl.col('dir') + 180) % 360)
        .otherwise(pl.col('dir'))
        .alias('dir'),
        pl.when(pl.col('playDirection') == 'left')
        .then((pl.col('o') + 180) % 360)
        .otherwise(pl.col('o'))
        .alias('o'),
    ])

    df = df.with_columns([
        ((pl.col('dir').cast(float) * math.pi / 180).cos() * pl.col('s')).alias('xs'),
        ((pl.col('dir').cast(float) * math.pi / 180).sin() * pl.col('s')).alias('ys')
    ])
    df = df.with_columns([
        pl.col("o").fill_null(0),
        pl.col("dir").fill_null(0),
        pl.col("xs").fill_null(0),
        pl.col("ys").fill_null(0),
    ])
    return df.sort(['gameId', 'playId', 'club', 'frameId'])

def get_presnap(df):
    return (
        df.filter(pl.col("frameType") == "BEFORE_SNAP")
    )

if __name__ == "__main__":
    import torch
    import math
    import argparse
    import polars as pl
    import numpy as np

    parser = argparse.ArgumentParser(description="Process football tracking data")
    parser.add_argument('--raw', action='store_true', help="Process data")
    parser.add_argument('--presnap', action='store_true', help="Process presnap data")
    parser.add_argument('--raw_OD', action='store_true', help="Process tensors")
    parser.add_argument('--presnap_OD', action='store_true', help="Process presnap data")
    parser.add_argument('--tensors', action='store_true', help="Process tensors")
    args = parser.parse_args()

    if args.raw or args.presnap:
        path_games = 'data/raw/games.csv'
        path_players = 'data/raw/players.csv'
        path_plays = 'data/raw/plays.csv'
        path_player_play = 'data/raw/player_play.csv'

        games = pl.read_csv(path_games, null_values=["NA"])
        players = pl.read_csv(path_players, null_values=["NA"])
        plays = pl.read_csv(path_plays, null_values=["NA"])
        player_play = pl.read_csv(path_player_play, null_values=["NA"])

        games = preprocess_games(games)
        players = preprocess_players(players)
        plays = preprocess_plays(plays)
        player_play = preprocess_player_play(player_play)

        tracking = []
        for week in range(1, 10):
            tracking_path = f"data/raw/tracking_week_{week}.csv"
            tracking_week = pl.read_csv(tracking_path, null_values=["NA"])
            tracking_week = preprocessing_tracking(tracking_week)
            if args.presnap or args.presnap_OD:
                tracking_week = get_presnap(tracking_week)
            tracking.append(tracking_week)

        tracking = pl.concat(tracking)

        df = (
            tracking
            .join(games, on=['gameId'], how='left')
            .join(players, on=['nflId'], how='left')
            .join(plays, on=['gameId', 'playId'], how='left')
            .join(player_play, on=['gameId', 'playId', 'nflId'], how='left')
        )

        week = df.filter(pl.col("week") == 1)
        if args.raw:
            week.write_csv("data/outputs/processed_week.csv")
            df.write_csv("data/outputs/processed.csv")
        if args.presnap:
            week.write_csv("data/outputs/presnap_week.csv")
            df.write_csv("data/outputs/presnap.csv")

    if args.presnap_OD or args.raw_OD:
        if args.raw_OD:
            df = pl.read_csv("data/outputs/processed.csv", null_values=["NA"])
        if args.presnap_OD:
            df = pl.read_csv("data/outputs/presnap.csv", null_values=["NA"])

        offense = df.filter(pl.col("club") != pl.col("defensiveTeam"))
        defense = df.filter(pl.col("club") != pl.col("possessionfullTeam"))
        offense_week = offense.filter(pl.col("week") == 1)
        defense_week = defense.filter(pl.col("week") == 1)
        if args.raw_OD:
            offense.write_csv("data/outputs/processed_offense.csv")
            defense.write_csv("data/outputs/processed_defense.csv")
            offense_week.write_csv("data/outputs/processed_offense_week.csv")
            defense_week.write_csv("data/outputs/processed_defense_week.csv")
        if args.presnap_OD:
            offense.write_csv("data/outputs/presnap_offense.csv")
            defense.write_csv("data/outputs/presnap_defense.csv")
            offense_week.write_csv("data/outputs/presnap_offense_week.csv")
            defense_week.write_csv("data/outputs/presnap_defense_week.csv")

    if args.tensors:
        def create_frame_tensor(df):
            def compute_relative_matrix(df, metric):
                values = df.select(pl.col(metric)).to_numpy().flatten()
                relative_matrix = values[:, None] - values
                return relative_matrix
            metrics = ['x', 'y', 's', 'a', 'dis', 'o', 'dir', 'xs', 'ys']
            matrices = [compute_relative_matrix(df, metric) for metric in metrics]
            stacked_matrix = np.stack(matrices, axis=-1)
            tensor = torch.tensor(stacked_matrix, dtype=torch.float32)
            return tensor

        tensors = {}
        for Id, frame_df in tracking.group_by(["Id"], maintain_order=True):
            frame_df.sort(["Id", 'club', "position"])
            tensors[Id] = create_frame_tensor(frame_df)