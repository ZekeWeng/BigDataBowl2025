# PREPROCESSING
import polars as pl
import math


def get_presnap(df):
    return (
        df.filter(pl.col("frameType") == "BEFORE_SNAP")
    )

def sort_id(df):
    return df.with_columns(
        pl.col('playId')
        .rank(method='dense')
        .over('gameId')
        .cast(pl.Int32)
        .alias('playId')
    )

def preprocess_games(df):
    return df.drop(['gameDate', 'gameTimeEastern', 'homeFinalScore', 'visitorFinalScore'])

def preprocess_players(df):
    return df.drop(['birthDate', 'collegeName', 'displayName'])

def preprocess_plays(df):
    df = sort_id(df)

    df = df.filter(
        (pl.col("qbSpike") != True) |
        (pl.col("qbKneel") != True) |
        (pl.col("qbSneak") != True) |
        (pl.col('rushLocationType') == "UNKNOWN") |
        (pl.col('passLocationType') == "UNKNOWN") |
        (pl.col("pff_passCoverage").is_not_null())
    ).with_columns(
        pl.when(pl.col('rushLocationType') == "INSIDE_RIGHT").then(pl.lit("RUN_INSIDE_RIGHT"))
        .when(pl.col('rushLocationType') == "INSIDE_LEFT").then(pl.lit("RUN_INSIDE_LEFT"))
        .when(pl.col('rushLocationType') == "OUTSIDE_RIGHT").then(pl.lit("RUN_OUTSIDE_RIGHT"))
        .when(pl.col('rushLocationType') == "OUTSIDE_LEFT").then(pl.lit("RUN_OUTSIDE_LEFT"))
        .when(pl.col('passLocationType') == "INSIDE_BOX").then(pl.lit("PASS_MIDDLE"))
        .when(pl.col('passLocationType') == "OUTSIDE_LEFT").then(pl.lit("PASS_OUTSIDE_LEFT"))
        .when(pl.col('passLocationType') == "OUTSIDE_RIGHT").then(pl.lit("PASS_OUTSIDE_RIGHT"))
        .otherwise(pl.lit("NA"))
        .alias('offensivePlay')
    ).with_columns(
        pl.when(pl.col('pff_runPassOption') == 1).then(pl.lit("RPO"))
        .otherwise(pl.col('offensivePlay'))
        .alias('offensivePlay')
    ).with_columns(
        pl.when(pl.col("pff_passCoverage") == "Cover-1 Double").then(pl.lit("Cover-1"))
        .when(pl.col("pff_passCoverage").is_in(
            ["Cover-3 Cloud Left", "Cover-3 Cloud Right", "Cover-3 Double Cloud", "Cover-3 Seam"])).then(pl.lit("Cover-3"))
        .when(pl.col("pff_passCoverage").is_in(["Cover 6-Left", "Cover-6 Right"])).then(pl.lit("Cover-6"))
        .when(pl.col("pff_passCoverage").is_in(["Miscellaneous", "None"])).then(pl.lit("Other"))
        .otherwise(pl.col("pff_passCoverage"))
        .alias("pff_passCoverage")
    )
    df = df.filter(
        (pl.col("pff_passCoverage").is_not_null()) &
        (pl.col("offensivePlay").is_not_null()) &
        (~pl.col("pff_passCoverage").is_in(["Prevent", "Bracket", "Other"])) &
        (pl.col("offensivePlay") != "NA")
    )

    columns_to_keep = [
        "gameId", "playId", "playDescription", "quarter", "down", "yardsToGo",
        "possessionTeam", "defensiveTeam", "preSnapHomeScore", "preSnapVisitorScore",
        "offensivePlay", "pff_passCoverage", "pff_manZone"
    ]
    df = df.select(columns_to_keep)

    return df.sort(["gameId", "playId"])

def preprocess_player_play(df):
    df = sort_id(df)
    columns_to_keep = [
        "gameId", "playId", "nflId",
        "inMotionAtBallSnap", "shiftSinceLineset", "motionSinceLineset",
        "pff_defensiveCoverageAssignment",
        "pff_primaryDefensiveCoverageMatchupNflId",
        "pff_secondaryDefensiveCoverageMatchupNflId"
    ]
    df = df.select(columns_to_keep)
    return df.sort(['gameId', 'playId'])

def preprocessing_tracking(df):
    df = sort_id(df)
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
    df = df.drop(["playDirection", "jerseyNumber"])
    return df.sort(['gameId', 'playId', 'club', 'frameId'])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process football tracking data")
    parser.add_argument('--raw', action='store_true', help="Process data")
    parser.add_argument('--presnap', action='store_true', help="Process presnap data")
    args = parser.parse_args()

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
        if args.presnap:
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

    df = df.filter(pl.col("pff_passCoverage").is_not_null())

    offense = (
        df.filter(pl.col("club") != pl.col("defensiveTeam"))
        .drop(["defensiveTeam", "possessionTeam"])
    )
    defense = (
        df.filter(pl.col("club") != pl.col("possessionTeam"))
        .drop(["defensiveTeam", "possessionTeam"])
    )

    def write_outputs(base_path, raw=False, presnap=False):
        if raw:
            df.write_csv(f"{base_path}/processed.csv")
            offense.write_csv(f"{base_path}/processed_offense.csv")
            defense.write_csv(f"{base_path}/processed_defense.csv")
        if presnap:
            df.write_csv(f"{base_path}/presnap.csv")
            offense.write_csv(f"{base_path}/presnap_offense.csv")
            defense.write_csv(f"{base_path}/presnap_defense.csv")

    write_outputs("data/outputs", raw=args.raw, presnap=args.presnap)
