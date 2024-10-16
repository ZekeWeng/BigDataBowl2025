import pandas as pd
import os

# Currently set to 2024
INPUT_PATH = "data/2025/raw"
OUTPUT_PATH = "data/2025/processed"

def process_week(week, games, plays, players, player_play):
    """
    Process the tracking data for a specific week.

    Params:
    - week: The week number to process.
    - games: DataFrame containing games data.
    - plays: DataFrame containing plays data.
    - players: DataFrame containing players data.

    Returns:
    - DataFrame for the processed week.
    """
    tracking = pd.read_csv(os.path.join(INPUT_PATH, f"tracking_week_{week}.csv"))
    games_plays = pd.merge(games[games.week == week], plays, how="inner", on="gameId")
    games_plays_tracking = pd.merge(games_plays, tracking, how="inner", on=["gameId", "playId"])
    games_plays_tracking_playerplays = pd.merge(games_plays_tracking, player_play, how="left", on = ["gameId", 'playId', 'nflId'])
    df = pd.merge(games_plays_tracking_playerplays, players, how="left", on = ["nflId", 'displayName'])

    output_file = f'{OUTPUT_PATH}/weeks/week_{week}.csv'
    df.to_csv(output_file, index=False)
    print(f"Processed data for week {week} saved to {output_file}.")

    return df


def process_games(df, week):
    """
    Converts processed weeks into a DataFrame for easy access

    Params:
    - df: The processed DataFrame for a week.
    - week: The week number.

    Returns:
    - None
    """
    for gameId in df.gameId.unique():
        game = df.loc[df.gameId==gameId].reset_index()
        play_ids = sorted(game['playId'].unique())
        playId_mapping = {playId: idx + 1 for idx, playId in enumerate(play_ids)}
        game['playId'] = game['playId'].map(playId_mapping)
        game = game.sort_values('playId')

        name = f"w{week}_{gameId}_{game.iloc[0]['visitorTeamAbbr']}@{game.iloc[0]['homeTeamAbbr']}"
        output_file = f'{OUTPUT_PATH}/games/{name}.csv'

        game.to_csv(output_file, index=False)
        print(f"Processd data for week {week}, game {name} saved to {output_file}.")


if __name__ == '__main__':
    directories = [
        INPUT_PATH,
        os.path.join(OUTPUT_PATH, "games"),
        os.path.join(OUTPUT_PATH, "weeks"),
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    games = pd.read_csv(os.path.join(INPUT_PATH, "games.csv"))
    plays = pd.read_csv(os.path.join(INPUT_PATH, "plays.csv"))
    players = pd.read_csv(os.path.join(INPUT_PATH, "players.csv"))
    player_play = pd.read_csv(os.path.join(INPUT_PATH, "player_play.csv"))
    all_weeks_data = []

    # Process each week
    for week in range(1, 2):
        week_data = process_week(week, games, plays, players, player_play)
        process_games(week_data, week)
        all_weeks_data.append(week_data)

    # Combine all weeks
    # df = pd.concat(all_weeks_data, ignore_index=True)
    # output_file = f'{OUTPUT_PATH}/processed_weeks.csv'
    # df.to_csv(output_file, index=False)
    # print(f"All weeks data combined and saved to {output_file}.")