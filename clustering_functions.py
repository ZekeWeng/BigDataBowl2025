import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pandas as pd
import math
from visualization import animate_play
from sklearn.cluster import SpectralClustering, KMeans
import numpy as np
import matplotlib.pyplot as plt
import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import pandas as pd


CONTROL_POINTS = 8
NUM_CLUSTERS = 10
RANDOM_STATE = 0


def add_movement_category(df):
    # Calculate moving average of speed
    df["s_moving_avg"] = df["s"].rolling(window=5).mean()

    # Identify the lineset frame for each play
    lineset_frames = df[df["event"] == "line_set"].groupby(["gameId", "playId"])["frameId"].min().reset_index()
    lineset_frames = lineset_frames.rename(columns={"frameId": "lineset_frame"})

    # Identify the maximum frame for each play
    max_frames = df.groupby(["gameId", "playId"])["frameId"].max().reset_index()
    max_frames = max_frames.rename(columns={"frameId": "max_frame"})

    # Merge the lineset frames and max frames back into the original DataFrame
    df = df.merge(lineset_frames, on=["gameId", "playId"], how="left")
    df = df.merge(max_frames, on=["gameId", "playId"], how="left")

    # Define conditions for start, end, and SNAP
    start_condition = (df["s_moving_avg"] > 0.8) & (df["s_moving_avg"].shift(1) <= 0.8) & (df["frameId"] > df["lineset_frame"])
    end_condition = (df["s_moving_avg"] <= 0.8) & (df["s_moving_avg"].shift(1) > 0.8) & (df["frameId"] > df["lineset_frame"])
    snap_condition = df["frameId"] == df["max_frame"]

    # Create a temporary column for movement type
    df["movementTypeTemp"] = np.select(
        [start_condition, end_condition, snap_condition],
        ["start", "end", "SNAP"],
        default=None
    )

    # Extract movement points
    movement_points = list(
        zip(df.loc[~df["movementTypeTemp"].isna(), "frameId"],
            df.loc[~df["movementTypeTemp"].isna(), "movementTypeTemp"])
    )

    # Initialize the movement column
    df["movement"] = "None"
    
    # Apply the logic to determine "shift" and "motion"
    for i in range(len(movement_points) - 1):  # Iterate until the second to last point
        cur = movement_points[i]
        next = movement_points[i + 1]
        if cur[1] == "start":
            if next[1] == "end":
                df.loc[(df["frameId"] >= cur[0]) & (df["frameId"] <= next[0]), "movement"] = "shift"
            elif next[1] == "SNAP":
                df.loc[(df["frameId"] >= cur[0]) & (df["frameId"] <= next[0]), "movement"] = "motion"

    # Handle the case where the last movement point is "start" and there is no "end" or "SNAP"
    if len(movement_points) > 0 and movement_points[-1][1] == "start":
        cur = movement_points[-1]
        df.loc[(df["frameId"] >= cur[0]) & (df["frameId"] <= df["max_frame"].max()), "movement"] = "motion"

    # Drop the temporary columns
    df = df.drop(columns=["lineset_frame", "max_frame"])

    # Return the processed group
    return df

def add_movement_column(df):
    # Initialize the movement column
    df["movement"] = None

    # Identify the frames where event is "man_in_motion" and "ball_snap"
    man_in_motion_frames = df[df["event"] == "man_in_motion"]
    ball_snap_frames = df[df["event"] == "ball_snap"]

    if ball_snap_frames.empty:
        return df

    # Merge the ball_snap_frames to get the snap frame for each play
    snap_frames = ball_snap_frames.groupby(["gameId", "playId"])["frameId"].min().reset_index()
    snap_frames = snap_frames.rename(columns={"frameId": "snap_frame"})
    df = df.merge(snap_frames, on=["gameId", "playId"], how="left")

    # Filter the man_in_motion_frames to only include frames before the snap
    # man_in_motion_frames = man_in_motion_frames[man_in_motion_frames["frameId"] < man_in_motion_frames["snap_frame"]]

    if man_in_motion_frames.empty:
        return df

    # Group by gameId and playId to find the player with the highest speed during "man_in_motion"
    max_speed_players = man_in_motion_frames.loc[man_in_motion_frames.groupby(["gameId", "playId"])["s"].idxmax()]

    # Mark the movement column for the player with the highest speed as "motion"
    for _, row in max_speed_players.iterrows():
        df.loc[(df["gameId"] == row["gameId"]) & (df["playId"] == row["playId"]) & (df["displayName"] == row["displayName"]) & (df["frameId"] >= row["frameId"]) & (df["frameId"] <= row["snap_frame"]), "movement"] = "motion"

    # Drop the temporary snap_frame column
    # df = df.drop(columns=["snap_frame"])

    return df



def plot_player_speed(df, gameId, playId, displayName=None):
    """
    Plot the movement of a specific player or all players in a specific play.

    Parameters:
        df (DataFrame): The dataframe containing the game data.
        gameId (int): The ID of the game.
        playId (int): The ID of the play.
        displayName (str, optional): The name of the player to plot. If None, plot all players.
    """
    # Filter the dataframe for the specific game and play
    filtered_data = df[(df["gameId"] == gameId) & (df["playId"] == playId)]
    
    # If displayName is provided, further filter for the specific player
    if displayName:
        filtered_data = filtered_data[filtered_data["displayName"] == displayName]

    if filtered_data.empty:
        if displayName:
            print(f"No data found for {displayName} in gameId {gameId}, playId {playId}.")
        else:
            print(f"No data found for gameId {gameId}, playId {playId}.")
        return

    # Define colors for each movement category
    movement_colors = {
        "None": "gray",  # Default color for None
        None: "gray",  # Default color for NaN
        "shift": "orange",
        "motion": "red"
    }

    # Create the plot
    plt.figure(figsize=(10, 6))

    # If plotting for multiple players, loop through each player's data
    for player_name, player_data in filtered_data.groupby("displayName"):
        # Sort by frameId for consistent plotting
        player_data = player_data.sort_values(by="frameId")
        
        # Plot the entire line, coloring based on the movement category
        for i in range(1, len(player_data)):
            # Get the current and previous rows
            current_row = player_data.iloc[i]
            previous_row = player_data.iloc[i - 1]

            # Plot the line segment between the current and previous rows
            plt.plot(
                [previous_row["frameId"], current_row["frameId"]],  # X-axis points
                [previous_row["s"], current_row["s"]],  # Y-axis points
                color=movement_colors[previous_row["movement"]],  # Use color of the previous row's movement
                linewidth=2,
                label=None if displayName else player_name  # Add a label only for all players
            )
        
        # Identify rows where movementTypeTemp is not NaN
        highlight_points = player_data[~player_data["movementTypeTemp"].isna()]

        # Add dots for the highlight points
        plt.scatter(
            highlight_points["frameId"], 
            highlight_points["s"], 
            label=None if displayName else f"{player_name} Movement Cut",
            zorder=5
        )

    # Manually create legend handles
    legend_handles = [
        Line2D([0], [0], color="red", lw=2, label="Motion (Red)"),
        Line2D([0], [0], color="orange", lw=2, label="Shift (Orange)"),
        Line2D([0], [0], color="gray", lw=2, label="None (Gray)")
    ]

    # Add the legend to the plot
    plt.legend(handles=legend_handles)

    # Add labels and title
    plt.ylim(0, 10)
    plt.xlabel("Frame ID")
    plt.ylabel("Speed (yds/s)")
    if displayName:
        plt.title(f"{displayName} Presnap Movement: Game {gameId}, Play {playId}")
        # plt.legend()
    else:
        plt.title(f"Presnap Player Speeds: Game {gameId}, Play {playId}")
    plt.show()


def animate_player_speed(play, displayName=None, output_gif=None):
    """
    Create an animated plot showing the movement of a specific player or all players in a specific play.

    Parameters:
        df (DataFrame): The dataframe containing the game data.
        gameId (int): The ID of the game.
        playId (int): The ID of the play.
        displayName (str, optional): The name of the player to animate. If None, animate all players.
        output_gif (str, optional): The file path to save the animation as a GIF. If None, the animation is not saved.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D
    import numpy as np

    # Filter the dataframe for the specific game and play
    filtered_data = play

    if displayName:
        filtered_data = filtered_data[filtered_data["displayName"] == displayName]

    if filtered_data.empty:
        print(f"No data found for gameId {play['gameId'].iloc[0]}, Play {play['playId'].iloc[0]}.")
        return

    movement_colors = {"None": "gray", None: "gray", "shift": "orange", "motion": "red"}

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_xlim(filtered_data["frameId"].min(), filtered_data["frameId"].max())
    ax.set_ylim(0, 10)
    ax.set_xlabel("Frame ID")
    ax.set_ylabel("Speed (yds/s)")
    ax.set_title(f"{displayName or 'All Players'} Presnap Movement: Game {play['gameId'].iloc[0]}, Play {play['playId'].iloc[0]}")

    segments = {player_name: [] for player_name in filtered_data["displayName"].unique()}
    scatter_points = []  # Store scatter point objects

    legend_handles = [
        Line2D([0], [0], color="red", lw=2, label="Motion (Red)"),
        Line2D([0], [0], color="orange", lw=2, label="Shift (Orange)"),
        Line2D([0], [0], color="gray", lw=2, label="None (Gray)"),
        Line2D([0], [0], color="blue", marker="o", linestyle="None", label="Movement Cut"),
    ]
    ax.legend(handles=legend_handles)

    def update(frame):
        for player_name in filtered_data["displayName"].unique():
            player_data = filtered_data[filtered_data["displayName"] == player_name]
            current_data = player_data[player_data["frameId"] <= frame]

            if current_data.shape[0] < 2:
                continue

            # Get the last two points to draw a segment
            last_two_points = current_data.iloc[-2:]
            x_values = last_two_points["frameId"].values
            y_values = last_two_points["s"].values
            movement_type = last_two_points["movement"].iloc[-1]
            color = movement_colors.get(movement_type, "gray")

            # Draw the segment
            line = ax.plot(x_values, y_values, color=color, lw=2, alpha=0.8)[0]
            segments[player_name].append(line)

            # Add a scatter point at the transition frame where the segment starts
            if len(segments[player_name]) > 1:
                prev_color = segments[player_name][-2].get_color()
                if color != prev_color:  # Detect a color change
                    scatter = ax.scatter(x_values[0], y_values[0], color="blue", zorder=5)
                    scatter_points.append(scatter)

        # Update scatter points for movementTypeTemp highlights
        highlight_data = filtered_data[
            (filtered_data["frameId"] == frame) & (filtered_data["movementTypeTemp"]=="SNAP")
        ]
        for _, point in highlight_data.iterrows():
            scatter = ax.scatter(point["frameId"], point["s"], color="blue", zorder=5)
            scatter_points.append(scatter)

        return [*scatter_points, *[line for seg_list in segments.values() for line in seg_list]]
    
    # Get the unique frame IDs and pad to 200 frames if necessary
    unique_frames = sorted(filtered_data["frameId"].unique())
    if len(unique_frames) < 200:
        unique_frames.extend([unique_frames[-1]] * (200 - len(unique_frames)))


    ani = FuncAnimation(fig, update, frames=unique_frames, interval=100, blit=True)

    if output_gif:
        ani.save(output_gif, writer=PillowWriter(fps=10))

    plt.show()



def standardize_to_fixed_frames(xs, ys, target_frames):
    """
    Standardize the number of frames in a play by subsampling the xs and ys lists.

    Parameters:
        xs (list or array): x-coordinates for all frames of a play.
        ys (list or array): y-coordinates for all frames of a play.
        target_frames (int): Number of frames to standardize to.

    Returns:
        np.ndarray: Concatenated array of standardized xs and ys.
    """
    # Convert to NumPy arrays for convenience
    xs = np.array(xs)
    ys = np.array(ys)

    # Handle case where there are fewer frames than target
    if len(xs) < target_frames:
        indices = np.linspace(0, len(xs) - 1, len(xs)).astype(int)
        xs = np.pad(xs[indices], (0, target_frames - len(xs)), 'edge')  # Pad with edge values
        ys = np.pad(ys[indices], (0, target_frames - len(ys)), 'edge')  # Pad with edge values
    else:
        # Subsample evenly spaced indices
        indices = np.linspace(0, len(xs) - 1, target_frames).astype(int)
        xs = xs[indices]
        ys = ys[indices]

    return np.concatenate([xs, ys])  # Concatenate xs and ys into a single array

NUM_CLUSTERS = 10
RANDOM_STATE = 0


def get_clusters(df, min_frames=CONTROL_POINTS, color = 'red', movement_type = 'Motion', num_clusters = NUM_CLUSTERS):

    df_sorted = df.sort_values(by=['gameId', 'playId', 'nflId', 'frameId'], ascending=[True, True, True, True])
    # df_sorted = df_sorted[df_sorted['position'].isin(['WR', 'TE', 'RB', 'FB', 'QB', 'T', 'G', 'C'])]
    # df_sorted

    movement = df_sorted.groupby(['gameId', 'playId', 'nflId']).filter(lambda x: len(x) >= min_frames)
    movement = movement.groupby(['gameId', 'playId', 'nflId'])

    movement_agg = movement.agg({
        'xs': list,
        'ys': list,
        'relative_x': list,
        'relative_y': list,
    }).reset_index()

    # Rename the columns for clarity
    movement_agg.rename(columns={'xs': 'x_vels', 'ys': 'y_vels'}, inplace=True)


    # Apply the function to standardize number of frames for each play
    movement_agg['vel_features'] = movement_agg.apply(lambda row: standardize_to_fixed_frames(row['x_vels'], row['y_vels'], target_frames=min_frames), axis=1)
    movement_agg['loc_features'] = movement_agg.apply(lambda row: standardize_to_fixed_frames(row['relative_x'], row['relative_y'], target_frames=min_frames), axis=1)
    movement_agg['features'] = movement_agg.apply(lambda row: np.concatenate([row['loc_features'], row['vel_features']]), axis=1)

    # movement_agg['features'] = movement_agg.apply(lambda row: standardize_to_fixed_frames(row['relative_x'], row['relative_y'], target_frames=min_frames), axis=1)


    # Step 3: Stack the feature vectors into a matrix
    features_matrix = np.stack(movement_agg['features'].values)


    # Apply Clustering
    # num_clusters = num_clusters
    # spectral = SpectralClustering(
    #     n_clusters=num_clusters,
    #     random_state=RANDOM_STATE,
    #     affinity='nearest_neighbors',  # Use 'rbf' or 'nearest_neighbors' based on your data
    #     n_neighbors=10  # Relevant only for 'nearest_neighbors' affinity
    # )

    spectral = KMeans(
        n_clusters=num_clusters,
        random_state=RANDOM_STATE,
        # affinity='nearest_neighbors',  # Use 'rbf' or 'nearest_neighbors' based on your data
        # n_neighbors=10  # Relevant only for 'nearest_neighbors' affinity
    )


    movement_agg['cluster'] = spectral.fit_predict(features_matrix)

    # --------- PLOT CLUSTERS ------------
    # Define the number of clusters and rows/columns for the grid
    rows = math.ceil(num_clusters / 3)  # Adjust the divisor to control the number of columns
    cols = 3

    # Create a grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))  # Adjust size per subplot
    axes = axes.flatten()  # Flatten the axes for easy iteration

    # Plot each cluster in a grid
    for cluster_id in range(num_clusters):
        ax = axes[cluster_id]  # Select the appropriate subplot
        cluster_plays = movement_agg[movement_agg['cluster'] == cluster_id]
        
        # Store all the motion lines for averaging
        all_xs = []
        all_ys = []

        for _, play in cluster_plays.head(40).iterrows():
            ys = play['features'][:min_frames]  # First min_frame values are ys
            xs = -1 * play['features'][min_frames:(2*min_frames)]  # Next min_frame values are xs
            ax.plot(xs, ys, marker='o', alpha=0.2, color='black')  # Individual motions in gray
            # ax.text(xs[-1], ys[-1], str((play['week'])), fontsize=6, alpha=0.7)
            all_xs.append(xs)
            all_ys.append(ys)
        
        # Calculate the average motion
        avg_xs = np.mean(all_xs, axis=0)
        avg_ys = np.mean(all_ys, axis=0)
        
        # Plot the average motion
        ax.plot(avg_xs, avg_ys, marker='o', alpha=1.0, color=color, linewidth=2, label=('Average ' + movement_type))

        # Calculate the direction vector from the second to last point to the last point
        dx = avg_xs[-1] - avg_xs[-2]
        dy = avg_ys[-1] - avg_ys[-2]

        # Normalize the direction vector
        length = np.sqrt(dx**2 + dy**2)
        dx /= length
        dy /= length

        # Calculate the starting point of the arrow
        arrow_end_x = avg_xs[-1] +  dx
        arrow_end_y = avg_ys[-1] +  dy

        # Add an arrow at the last marker
        ax.annotate(
            # '', xy=(avg_xs[-1], avg_ys[-1]), xytext=(arrow_start_x, arrow_start_y),
            '', xy=(arrow_end_x, arrow_end_y), xytext=(avg_xs[-1], avg_ys[-1]),

            arrowprops=dict(facecolor=color, shrink=0.05, width=2, headwidth=8)
        )

        
        ax.set_ylim(-10, 10)
        ax.set_xlim(-25, 25)
        ax.set_title(f'Cluster {cluster_id}')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8)

    # Remove any empty subplots
    for i in range(num_clusters, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle((movement_type + " Clusters"), fontsize=16, y=1.02)  # Adjust y to control vertical position
    plt.tight_layout()
    plt.show()



    return movement_agg[['gameId', 'playId', 'nflId', 'cluster']]


def get_movement(offense):
    
  movement = offense[offense["frameType"].isin(["BEFORE_SNAP", "SNAP"]) & offense["position"].isin(["RB", "FB", "WR", "TE"])]

  movement = (
      movement
      .sort_values(by=["week", "gameId", "playId", "frameId"])  # Sort the entire dataframe first
      .groupby(["gameId", "playId", "displayName"])
      .apply(add_movement_category)
      .reset_index(drop=True)
  )

  shifts = movement[movement["movement"] == "shift"]
  motions = movement[movement["movement"] == "motion"]


  # shifts.to_csv('shifts.csv')
  # motions.to_csv('motions.csv')


  motion_clusters = get_clusters(motions)
  shift_clusters = get_clusters(shifts)


  # Step 1: Map motion labels
  motion_cluster_mapping = {
      0: "Jet",  # right
      1: "Fly",  # left
      2: "Fly",  # right
      3: "Jet",  # left
      4: "Glide",  # left
      5: "Orbit",  # right
      6: "Glide",  # right
      7: "Over",  # left
      8: "Orbit",  # left
      9: "Over"  # right
  }
  motion_clusters['cluster_label'] = motion_clusters['cluster'].map(motion_cluster_mapping)


  # Step 3: Merge motion clusters
  motion_merged = movement.merge(
      motion_clusters,
      on=['week', 'gameId', 'playId', 'displayName'],
      how='left'
  )

  # Step 4: Add motion labels conditionally
  movement['movementClassification'] = motion_merged.apply(
      lambda row: row['cluster_label'] if row['movement'] == 'motion' else None,
      axis=1
  )


  # Step 2: Map shift labels
  shift_cluster_mapping = {
      0: "Shift RB in backfield",  # to left of QB
      1: "Shift backfield to inline",  # to LOS left
      2: "Shift backfield to inline",  # to LOS right
      3: "Shift across field",  # to left
      4: "Shift across field",  # to right
      5: "Shift inward",  # from left
      6: "Shift RB in backfield",  # to right of QB
      7: "Shift backfield out",  # to right
      8: "Shift backfield out",  # to left
      9: "Shift inward"  # from right

  }
  shift_clusters['cluster_label'] = shift_clusters['cluster'].map(shift_cluster_mapping)

  # Step 5: Merge shift clusters
  shift_merged = movement.merge(
      shift_clusters,
      on=['week', 'gameId', 'playId', 'displayName'],
      how='left'
  )

  # Step 6: Add shift labels without overwriting existing motion labels
  movement['movementClassification'] = shift_merged.apply(
      lambda row: row['cluster_label'] if row['movement'] == 'shift' and pd.isna(movement.loc[row.name, 'movementClassification']) else movement.loc[row.name, 'movementClassification'],
      axis=1
  )


  offense_columns = offense.columns.tolist()

  merged = offense.merge(movement, on=offense_columns, how='left')
  return merged

#   merged.to_csv('movement.csv')

motions = (
    offense
    .sort(["week", "gameId", "playId", "nflId", "frameId"])
)
motions = motions.with_columns([
    pl.col("s")
        .rolling_mean(window_size=4, min_periods=1)
        .over(["gameId", "playId", "nflId"])
        .alias("s_moving_avg"),
    pl.col("dis")
        .rolling_sum(window_size=4, min_periods=1)
        .over(["gameId", "playId", "nflId"])
        .alias("dis_moving_sum")
])
motions = motions.with_columns((
    (pl.col("s_moving_avg") > 0.62) &
    (pl.col("dis_moving_sum") > 1.2))
        .over(["gameId", "playId", "nflId"])
        .cast(pl.Int64)
        .alias("movement_condition")
).with_columns([
    (pl.col("movement_condition")
        .tail(6)
        .sum()
        .over(["gameId", "playId", "nflId"])
        .alias("motion_exists")),
]).filter(pl.col("motion_exists") > 0).sort(["gameId", "playId", "nflId", "frameId"], descending=True)

motions = motions.with_columns(
    (pl.col("movement_condition"))
        .cum_sum()
        .over(["gameId", "playId", "nflId"])
        .alias("motion_cs")
).with_columns([
    pl.when(
        (pl.col("motion_cs") == 0) |
        (pl.col("motion_cs") == pl.col("motion_cs").shift(-1)-1))
        .then(1)
        .otherwise(0)
        .over(["gameId", "playId", "nflId"])
        .alias("motion")
])
motions = motions.with_columns(
    pl.col("motion")
        .rolling_max(window_size=4, min_periods=1)
        .over(["gameId", "playId", "nflId"])
        .alias("motion")
).filter(pl.col("motion")==1).sort(["gameId", "playId", "nflId", "frameId"])

motions.select(["gameId", "playId"]).unique().sort("gameId", "playId")#[10:20]