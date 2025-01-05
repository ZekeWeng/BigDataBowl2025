################################################################################
#
# Preprocesssing
#
################################################################################

import polars as pl
import math

schema = {
    # ID
    "gameId": pl.Int32,
    "playId": pl.Int32,
    "frameId": pl.Int32,
    "club": pl.Categorical,
    "nflId": pl.Int32,
    "displayName": pl.Categorical,

    # Tracking
    "x": pl.Float32,
    "y": pl.Float32,
    "relative_x": pl.Float32,
    "relative_y":  pl.Float32,
    "xs":  pl.Float32,
    "ys":  pl.Float32,
    "s":  pl.Float32,
    "xa":  pl.Float32,
    "ya":  pl.Float32,
    "a":  pl.Float32,
    "dis":  pl.Float32,
    "o":  pl.Float32,
    "dir":  pl.Float32,

    # Context
    "yardsGained":  pl.Float32,
    "score_margin":  pl.Float32,
    "yardsToGo": pl.Float32,
    "down": pl.Int32,
    "quarter": pl.Int32,
    "playDescription": pl.String,
    "absoluteYardlineNumber": pl.Int32,
    "gameClock": pl.String,
    "playDirection": pl.Categorical,
    "week": pl.Int32,
    "time": pl.String,

    # MISC
    "event": pl.Categorical,
    "possessionTeam": pl.Categorical,
    "defensiveTeam": pl.Categorical,
    "frameType": pl.Categorical,
    "homeTeamAbbr": pl.Categorical,
    "visitorTeamAbbr": pl.Categorical,

    # Player
    "position": pl.Categorical,
    "height": pl.Categorical,
    "weight": pl.Int32,
    "jerseyNumber": pl.Int32,

    # PFF
    "pff_manZone": pl.Categorical,
    "pff_passCoverage": pl.Categorical,
    "defCoverage": pl.Categorical,
    "runPass": pl.Categorical,
    "inMotionAtBallSnap": pl.Int32,
    "motionSinceLineset": pl.Int32,
    "shiftSinceLineset": pl.Int32,
    "pff_primaryDefensiveCoverageMatchupNflId": pl.Int32,
    "pff_secondaryDefensiveCoverageMatchupNflId": pl.Int32,
    "wasTargettedReceiver": pl.Int32,
}

files = {
    "games": "/Users/zekeweng/Dropbox/BigDataBowl/kaggle/games.csv",
    "players": "/Users/zekeweng/Dropbox/BigDataBowl/kaggle/players.csv",
    "plays": "/Users/zekeweng/Dropbox/BigDataBowl/kaggle/plays.csv",
    "player_play": "/Users/zekeweng/Dropbox/BigDataBowl/kaggle/player_play.csv",
}

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
    df = df.with_columns(
        pl.col("qbSpike").fill_null(False).alias("qbSpike"),
        pl.col("qbSneak").fill_null(False).alias("qbSneak"),
        pl.col("qbKneel").cast(pl.Boolean).alias("qbKneel"),
        pl.col("pff_runPassOption").cast(pl.Boolean).alias("pff_runPassOption"),
    )
    df = df.filter(
            (pl.col("qbSpike") == False) &
            (pl.col("qbKneel") == False) &
            (pl.col("qbSneak") == False) &
            (pl.col('pff_runPassOption') == False) &
            (pl.col("pff_passCoverage").is_not_null())
        )
    # Redefining Run / Pass
    df = df.with_columns(
        pl.when(pl.col('rushLocationType').is_not_null())
        .then(pl.lit("RUN"))
        .when(pl.col('passLocationType').is_not_null())
        .then(pl.lit("PASS"))
        .otherwise(None)
        .alias('runPass')
    )
    # Redefining defCoverage
    df = df.with_columns([
        pl.when(pl.col("pff_passCoverage") == "Cover-1 Double").then(pl.lit("Cover-1"))
            .when(pl.col("pff_passCoverage").is_in(["Cover-3 Cloud Left", "Cover-3 Cloud Right", "Cover-3 Double Cloud", "Cover-3 Seam"])).then(pl.lit("Cover-3"))
            # .when(pl.col("pff_passCoverage").is_in(["Cover 6-Left", "Cover-6 Right"])).then(pl.lit("Cover-6"))
            .when(pl.col("pff_passCoverage").is_in(["2-Man", "Bracket", "Cover-0", "Cover 6-Left", "Cover-6 Right", "Goal Line", "Miscellaneous", "None", "Prevent", "Red Zone"])).then(None)
            .otherwise(pl.col("pff_passCoverage"))
            .alias("defCoverage")
    ])
    # Cut Nones
    df = df.filter(
        (pl.col("defCoverage").is_not_null()) &
        (pl.col("runPass").is_not_null())
    )
    # Creating
    df = df.select([
        "gameId", "playId", "playDescription", "quarter", "down", "yardsToGo",
        "possessionTeam", "defensiveTeam", "preSnapHomeScore", "preSnapVisitorScore",
        "runPass", "defCoverage", "pff_passCoverage", "pff_manZone", "absoluteYardlineNumber", "gameClock",
        "yardsGained"
    ])
    return df

def preprocess_player_play(df):
    df = sort_id(df)
    df = df.with_columns([
        pl.when(pl.col("inMotionAtBallSnap") == True)
        .then(1)
        .otherwise(0)
        .cast(pl.Int64)
        .alias("inMotionAtBallSnap"),
        pl.when(pl.col("motionSinceLineset") == True)
        .then(1)
        .otherwise(0)
        .cast(pl.Int64)
        .alias("motionSinceLineset"),
        pl.when(pl.col("shiftSinceLineset") == True)
        .then(1)
        .otherwise(0)
        .cast(pl.Int64)
        .alias("shiftSinceLineset"),
        pl.when(pl.col("pff_primaryDefensiveCoverageMatchupNflId").is_not_null())
        .then(pl.col("pff_primaryDefensiveCoverageMatchupNflId"))
        .otherwise(0)
        .cast(pl.Int64)
        .alias("pff_primaryDefensiveCoverageMatchupNflId")
    ])
    columns_to_keep = [
        "gameId", "playId", "nflId",
        "inMotionAtBallSnap", "shiftSinceLineset", "motionSinceLineset",
        "pff_defensiveCoverageAssignment",
        "pff_primaryDefensiveCoverageMatchupNflId",
        "pff_secondaryDefensiveCoverageMatchupNflId",
        "wasTargettedReceiver"
    ]
    df = df.select(columns_to_keep)
    return df

def preprocessing_tracking(df):
    df = sort_id(df)
    # Correcting Direction
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
    # Join ball data
    ball_data = df.filter(pl.col("displayName") == "football").rename({"x": "x_football", "y": "y_football"})
    df = df.join(
        ball_data.select(["gameId", "playId", "frameId", "x_football", "y_football"]),
        on=["gameId", "playId", "frameId"],
        how="inner"
    ).with_columns([
        (pl.col("x") - pl.col("x_football")).alias("relative_x"),
        (pl.col("y") - pl.col("y_football")).alias("relative_y"),
        ((pl.col('dir').cast(float) * math.pi / 180).cos() * pl.col('s')).alias('xs'),
        ((pl.col('dir').cast(float) * math.pi / 180).sin() * pl.col('s')).alias('ys'),
        ((pl.col('dir').cast(float) * math.pi / 180).cos() * pl.col('a')).alias('xa'),
        ((pl.col('dir').cast(float) * math.pi / 180).sin() * pl.col('a')).alias('ya')
    ]).with_columns([
        pl.col("o").fill_null(0),
        pl.col("dir").fill_null(0),
        pl.col("xs").fill_null(0),
        pl.col("ys").fill_null(0),
    ])
    if PRESNAP:
        df = df.filter(pl.col("displayName") != "football")
    return df

def cut_and_trim(df):
    df_frame_limit = (
        df
        .group_by(["gameId", "playId"])
        .agg(pl.col("frameId").max().alias("max_frameId"))
        .filter(pl.col("max_frameId") <= 256)
    )

    df = df.join(df_frame_limit, on=["gameId", "playId"], how="inner")
    return df

def preprocess(week, df_plays):
    tracking_week = pl.read_csv(get_file(f"tracking_{week}"), null_values=["NA"])
    df = preprocessing_tracking(tracking_week)

    if PRESNAP:
        df = df.filter(pl.col("frameType") == "BEFORE_SNAP")
        df = cut_and_trim(df)
        df = (
            df.join(df_plays, on=['gameId', 'playId', 'nflId'], how='inner')
        )
    else:
        df = (df.join(df_plays, on=['gameId', 'playId', 'nflId'], how='left'))
    df = (
        df.with_columns(
            pl.when(pl.col('homeTeamAbbr') == pl.col('club'))
            .then(pl.col('preSnapHomeScore') - pl.col('preSnapVisitorScore'))
            .otherwise(pl.col('preSnapVisitorScore') - pl.col('preSnapHomeScore'))
            .alias('score_margin'),
        )
        .with_columns(
            pl.when(pl.col('playDirection') == 'left')
            .then((100 - pl.col('absoluteYardlineNumber')))
            .otherwise(pl.col('absoluteYardlineNumber'))
            .alias('absoluteYardlineNumber'),
        )
    )
    zone_filter = (
        df.filter(
            (pl.col("pff_manZone") == "Zone") &
            (pl.col("inMotionAtBallSnap") == 0) &
            (pl.col("motionSinceLineset") == 0) &
            (pl.col("shiftSinceLineset") == 0))
        .group_by(["gameId", "playId"], maintain_order=True)
            .agg(pl.col("frameId")
            .max()
            .alias("playLength"))
    )
    zone_removal_ids = zone_filter.filter(pl.col("playLength") <= 100).select(["gameId", "playId"])
    df = df.join(
        zone_removal_ids,
        on=["gameId", "playId"],
        how="anti"
    )
    df = df.select(schema.keys())

    return df

def get_file(data):
    dropbox_path = files[data]
    return dropbox_path

def write_outputs(df, file_name):
    if PRESNAP:
        path = f"/Users/zekeweng/Dropbox/BigDataBowl/presnap/{file_name}.csv"
    else:
        path = f"/Users/zekeweng/Dropbox/BigDataBowl/plays/{file_name}.csv"
    df.write_csv(path)
    print(f"Wrote {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process some football tracking data.")
    parser.add_argument("--presnap", action="store_true", help="Process raw data")
    args = parser.parse_args()
    PRESNAP = args.presnap

    for week in range(1, 10):
        files[f"tracking_{week}"] = f"/Users/zekeweng/Dropbox/BigDataBowl/kaggle/tracking_week_{week}.csv"

    games = preprocess_games(pl.read_csv(get_file("games"), null_values=["NA"]))
    players = preprocess_players(pl.read_csv(get_file("players"), null_values=["NA"]))
    plays = preprocess_plays(pl.read_csv(get_file("plays"), null_values=["NA"]))
    player_play = preprocess_player_play(pl.read_csv(get_file("player_play"), null_values=["NA"]))

    df_plays = (
        plays
        .join(player_play, on=['gameId', 'playId'], how='left')
        .join(players, on=['nflId'], how='left')
        .join(games, on=['gameId'], how='left')
    )

    for week in range(1,10):
        df = preprocess(week, df_plays)
        write_outputs(df, f"week_{week}")