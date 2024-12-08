################################################################################
#
# Preprocesssing
#
################################################################################

import polars as pl
import math

column_order = [
    # ID
    "gameId",  # Game Identifier
    "playId",  # Play Identifier
    "frameId",  # Frame Identifier
    "club",  # Team Club
    "nflId",  # NFL Player Identifier
    "displayName",  # Player Name

    # Tracking
    "x",  # X-coordinate
    "y",  # Y-coordinate
    "relative_x",  # Relative X-coordinate
    "relative_y",  # Relative Y-coordinate
    "xs",  # X-speed
    "ys",  # Y-speed
    "s",  # Speed
    "a",  # Acceleration
    "dis",  # Distance
    "o",  # Orientation
    "dir",  # Direction

    # Context
    "yardsGained",  # Yards Gained
    "score_margin",  # Score Margin
    "yardsToGo",  # Yards to Go
    "yardsToGoCat",  # Categorized Yards to Go
    "down",  # Down
    "quarter",  # Quarter
    "playDescription",  # Play Description
    "absoluteYardlineNumber",  # Absolute Yardline Number
    "gameClock",  # Game Clock
    "playDirection",  # Play Direction
    "week",  # Week
    "time",  # Time of Play

    # MISC
    "event",  # Event
    "possessionTeam",  # Possession Team
    "defensiveTeam",  # Defensive Team
    "frameType",  # Frame Type

    # Player
    "position",  # Player Position
    "height",  # Player Height
    "weight",  # Player Weight

    # PFF
    "pff_manZone",  # Man or Zone Coverage
    "pff_passCoverage",  # pff's Pass Coverage
    "passCoverage",  # Pass Coverage Type
    "offensivePlay",  # Offensive Play Type
    "inMotionAtBallSnap",  # In Motion at Ball Snap
    "motionSinceLineset",  # Motion Since Line Set
    "shiftSinceLineset",  # Shift Since Line Set
    "pff_primaryDefensiveCoverageMatchupNflId",  # Primary Defensive Coverage Matchup
    "pff_secondaryDefensiveCoverageMatchupNflId",  # Secondary Defensive Coverage Matchup
    "wasTargettedReceiver",  # Targeted Receiver
    "pff_defensiveCoverageAssignment",  # Defensive Coverage Assignment

    # Offensive Positions
    "QB",  # Quarterback
    "RB",  # Running Back
    "FB",  # Fullback
    "WR",  # Wide Receiver
    "TE",  # Tight End
    "OL",   # Center + Guard + Tackle

    # Defensive Positions
    "DE",  # Defensive End
    "DT",  # Defensive Tackle + Nose Tackle
    "LB",  # Middle Linebacker + Outside Linebacker + Inside Linebacker
    "CB",  # Cornerback + Defensive Back
    "S",  # Free Safety + Strong Safety
]

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
    df = df.with_columns(
        pl
        .when(pl.col("position") == "NT").then(pl.lit("DT"))
        .when(pl.col("position") == "ILB").then(pl.lit("LB"))
        .when(pl.col("position") == "MLB").then(pl.lit("LB"))
        .when(pl.col("position") == "OLB").then(pl.lit("LB"))
        .when(pl.col("position") == "FS").then(pl.lit("S"))
        .when(pl.col("position") == "SS").then(pl.lit("S"))
        .when(pl.col("position") == "DB").then(pl.lit("CB"))
        .when(pl.col("position") == "C").then(pl.lit("OL"))
        .when(pl.col("position") == "G").then(pl.lit("OL"))
        .when(pl.col("position") == "T").then(pl.lit("OL"))
        .otherwise(pl.col("position"))
        .alias("position")
    )
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
    ).with_columns([
        pl.when(pl.col('rushLocationType') == "INSIDE_RIGHT").then(pl.lit("RUN_INSIDE_RIGHT"))
         .when(pl.col('rushLocationType') == "INSIDE_LEFT").then(pl.lit("RUN_INSIDE_LEFT"))
         .when(pl.col('rushLocationType') == "OUTSIDE_RIGHT").then(pl.lit("RUN_OUTSIDE_RIGHT"))
         .when(pl.col('rushLocationType') == "OUTSIDE_LEFT").then(pl.lit("RUN_OUTSIDE_LEFT"))
         .when(pl.col('passLocationType') == "INSIDE_BOX").then(pl.lit("PASS_MIDDLE"))
         .when(pl.col('passLocationType') == "OUTSIDE_LEFT").then(pl.lit("PASS_OUTSIDE_LEFT"))
         .when(pl.col('passLocationType') == "OUTSIDE_RIGHT").then(pl.lit("PASS_OUTSIDE_RIGHT"))
         .otherwise(pl.lit("NA"))
         .alias('offensivePlay')
    ])
    df = df.with_columns([
        pl.when(pl.col('pff_runPassOption') == 1).then(pl.lit("RPO"))
         .otherwise(pl.col('offensivePlay'))
         .alias('offensivePlay'),
        pl.when(pl.col("pff_passCoverage") == "Cover-1 Double").then(pl.lit("Cover-1"))
         .when(pl.col("pff_passCoverage").is_in(["Cover-3 Cloud Left", "Cover-3 Cloud Right", "Cover-3 Double Cloud", "Cover-3 Seam"])).then(pl.lit("Cover-3"))
         .when(pl.col("pff_passCoverage").is_in(["Cover 6-Left", "Cover-6 Right"])).then(pl.lit("Cover-6"))
         .when(pl.col("pff_passCoverage").is_in(["Miscellaneous", "None"])).then(pl.lit("Other"))
         .otherwise(pl.col("pff_passCoverage"))
         .alias("passCoverage")
    ]).filter(
        (pl.col("passCoverage").is_not_null()) &
        (pl.col("offensivePlay").is_not_null()) &
        (~pl.col("passCoverage").is_in(["Prevent", "Bracket", "Other"])) &
        (pl.col("offensivePlay") != "NA")
    ).with_columns(
        pl.when(pl.col("yardsToGo") <= 3).then(pl.lit("Short"))
         .when((pl.col("yardsToGo") >= 4) & (pl.col("yardsToGo") <= 8)).then(pl.lit("Medium"))
         .when((pl.col("yardsToGo") >= 9) & (pl.col("yardsToGo") <= 12)).then(pl.lit("Long"))
         .otherwise(pl.lit("Very Long"))
         .alias("yardsToGoCat")
    ).select([
        "gameId", "playId", "playDescription", "quarter", "down", "yardsToGo", "yardsToGoCat",
        "possessionTeam", "defensiveTeam", "preSnapHomeScore", "preSnapVisitorScore",
        "offensivePlay", "passCoverage", "pff_passCoverage", "pff_manZone", "absoluteYardlineNumber", "gameClock",
        "yardsGained"
    ])
    return df

def preprocess_player_play(df):
    df = sort_id(df)
    df = df.with_columns([
        pl.when(pl.col("inMotionAtBallSnap") == True)
        .then(0)
        .otherwise(0)
        .alias("inMotionAtBallSnap"),
        pl.when(pl.col("motionSinceLineset") == True)
        .then(0)
        .otherwise(0)
        .alias("motionSinceLineset"),
        pl.when(pl.col("shiftSinceLineset") == True)
        .then(0)
        .otherwise(0)
        .alias("shiftSinceLineset"),
        pl.when(pl.col("pff_primaryDefensiveCoverageMatchupNflId").is_null())
        .then(0)
        .otherwise(pl.col("pff_primaryDefensiveCoverageMatchupNflId"))
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
        ((pl.col('dir').cast(float) * math.pi / 180).sin() * pl.col('s')).alias('ys')
    ]).with_columns([
        pl.col("o").fill_null(0),
        pl.col("dir").fill_null(0),
        pl.col("xs").fill_null(0),
        pl.col("ys").fill_null(0),
    ]).drop(["jerseyNumber"]).filter(pl.col("displayName") != "football")
    return df

def cut_and_trim(df):
    df_frame_limit = (
        df
        .group_by(["gameId", "playId"])
        .agg(pl.col("frameId").max().alias("max_frameId"))
        .filter(pl.col("max_frameId") < 250)
    )

    df = df.join(df_frame_limit, on=["gameId", "playId"], how="inner")
    df = df.filter(pl.col("frameId") % 2 == 1)
    return df

def create_position_count(df):
    position_counts = df.with_columns(pl.lit(1).alias("count"))
    position_counts = position_counts.pivot(
        values="count",
        index=["gameId", "playId", "frameId"],
        on="position",
        aggregate_function="sum",
    ).fill_null(0)
    return df.join(position_counts, on=['gameId', 'playId', 'frameId'], how='inner')

def preprocess(week, games, players, plays, player_play):
    tracking_week = pl.read_csv(get_file(f"tracking_{week}"), null_values=["NA"])
    df = preprocessing_tracking(tracking_week)

    if PRESNAP:
        df = df.filter(pl.col("frameType") != "AFTER_SNAP")
        df = cut_and_trim(df)

    df = (
        df
        .join(plays, on=['gameId', 'playId'], how='left')
        .join(player_play, on=['gameId', 'playId', 'nflId'], how='left')
        .join(players, on=['nflId'], how='left')
        .join(games, on=['gameId'], how='left')
    )
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
    df = create_position_count(df)
    df = df.select(column_order)

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

    for week in range(1,10):
        df = preprocess(week, games, players, plays, player_play)
        write_outputs(df, f"week_{week}")