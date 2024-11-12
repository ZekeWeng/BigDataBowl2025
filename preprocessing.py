import argparse
import CONSTANTS
import polars as pl

def preprocess(df_tracking):
    df = df_tracking.with_columns([
        (pl.col('club') == pl.col('possessionTeam')).alias('is_on_offense'),
        (pl.col('club') == pl.col('defensiveTeam')).alias('is_on_defense')
    ])
    return df

def preprocess_plays(df_plays, df_player_play):
    df = df_plays.join(
      df_player_play,
      on=['gameId', 'playId'],
      how='left'
    )

    df = df.with_columns([
        pl.col('playId')
        .rank(method='dense')
        .over('gameId')
        .cast(pl.Int32)
        .alias('playId'),
    ])
    df = df.with_columns([
        (pl.col('gameId').cast(pl.Utf8) + '_' + pl.col('playId').cast(pl.Utf8)).alias('Id')
    ])

    df = df.sort(['gameId', 'playId'])

    return df

def preprocess_tracking(df_tracking):
    df = df_tracking.with_columns([
        pl.col('playId')
        .rank(method='dense')
        .over('gameId')
        .cast(pl.Int32)
        .alias('playId'),
    ])

    df = df.with_columns([
        (pl.col('gameId').cast(pl.Utf8) + '_' + pl.col('playId').cast(pl.Utf8)).alias('Id')
    ])

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

    df = df.sort(['gameId', 'playId', 'frameId', 'club'])
    return df

def preprocess_presnap(df):
    line_set_frames = (
        df.filter(pl.col('event') == 'line_set')
          .group_by(['gameId', 'playId'])
          .agg([pl.col('frameId').min().alias('frameId_line_set')])
    )

    snap_frames = (
        df.filter(pl.col('frameType') == 'SNAP')
          .group_by(['gameId', 'playId'])
          .agg([pl.col('frameId').min().alias('frameId_snap')])
    )

    valid_frames = line_set_frames.join(snap_frames, on=['gameId', 'playId'], how='inner')
    valid_frames = valid_frames.filter(pl.col('frameId_line_set') <= pl.col('frameId_snap'))

    df = df.join(valid_frames, on=['gameId', 'playId'], how='inner')
    df = df.filter(
        (pl.col('frameId') >= pl.col('frameId_line_set')) &
        (pl.col('frameId') <= pl.col('frameId_snap'))
    )

    df = df.drop(['frameId_line_set', 'frameId_snap'])
    df = df.sort(['gameId', 'playId', 'frameId', 'club', 'nflId'])

    return df

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process football tracking data.")
    parser.add_argument('--process', action='store_true', help="Process the main 'processed' data")
    parser.add_argument('--presnap', action='store_true', help="Process the 'presnap' data")
    args = parser.parse_args()

    # Load data
    games = pl.read_csv(CONSTANTS.GAMES)
    players = pl.read_csv(CONSTANTS.PLAYERS)
    plays = pl.read_csv(CONSTANTS.PLAYS, null_values=["NA"])
    player_play = pl.read_csv(CONSTANTS.PLAYER_PLAY)

    plays_with_player_play = preprocess_plays(plays, player_play)

    if args.process:
        processed = []
        for week in range(1, 10):
            tracking = pl.read_csv(f"{CONSTANTS.INPUT}/tracking_week_{week}.csv", null_values=["NA"])
            tracking = preprocess_tracking(tracking)

            processed_week = tracking.join(plays_with_player_play, on=['Id', 'nflId', 'gameId', 'playId'], how='left')
            processed_week = (
                processed_week
                .join(games, on=['gameId'], how='left')
                .join(players, on=['nflId', 'displayName'], how='left')
            )
            processed.append(processed_week)

        df_processed = pl.concat(processed)
        df_processed = df_processed.filter(pl.col('yardsGained').is_not_null())
        df_processed.write_csv(f"{CONSTANTS.OUTPUT}/processed.csv")

    if args.presnap:
        presnap = []
        for week in range(1, 10):
            tracking = pl.read_csv(f"{CONSTANTS.INPUT}/tracking_week_{week}.csv", null_values=["NA"])
            tracking = preprocess_tracking(tracking)
            presnap_tracking = preprocess_presnap(tracking)

            presnap_week = presnap_tracking.join(plays_with_player_play, on=['Id', 'nflId', 'gameId', 'playId'], how='left')
            presnap_week = (
                presnap_week
                .join(games, on=['gameId'], how='left')
                .join(players, on=['nflId', 'displayName'], how='left')
            )
            presnap.append(presnap_week)

        df_presnap = pl.concat(presnap)
        df_presnap = df_presnap.filter(pl.col('yardsGained').is_not_null())
        df_presnap.write_csv(f"{CONSTANTS.OUTPUT}/presnap.csv")