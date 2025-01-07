"""
Acknowledgment:
This implementation is based on the original work by Nick Wan.

Modifications have been made to adapt the code to meet specific project requirements.

Original source: https://www.kaggle.com/code/nickwan/animate-plays-with-plotly-real-no-lies-here
"""

import plotly.graph_objects as go
import numpy as np
import imageio
import os



COLORS = {
    'ARI':["#97233F","#000000","#FFB612"],
    'ATL':["#A71930","#000000","#A5ACAF"],
    'BAL':["#241773","#000000"],
    'BUF':["#00338D","#C60C30"],
    'CAR':["#0085CA","#101820","#BFC0BF"],
    'CHI':["#0B162A","#C83803"],
    'CIN':["#FB4F14","#000000"],
    'CLE':["#311D00","#FF3C00"],
    'DAL':["#003594","#041E42","#869397"],
    'DEN':["#FB4F14","#002244"],
    'DET':["#0076B6","#B0B7BC","#000000"],
    'GB' :["#203731","#FFB612"],
    'HOU':["#03202F","#A71930"],
    'IND':["#002C5F","#A2AAAD"],
    'JAX':["#101820","#D7A22A","#9F792C"],
    'KC' :["#E31837","#FFB81C"],
    'LA' :["#003594","#FFA300","#FF8200"],
    'LAC':["#0080C6","#FFC20E","#FFFFFF"],
    'LV' :["#000000","#A5ACAF"],
    'MIA':["#008E97","#FC4C02","#005778"],
    'MIN':["#4F2683","#FFC62F"],
    'NE' :["#002244","#C60C30","#B0B7BC"],
    'NO' :["#101820","#D3BC8D"],
    'NYG':["#0B2265","#A71930","#A5ACAF"],
    'NYJ':["#125740","#000000","#FFFFFF"],
    'PHI':["#004C54","#A5ACAF","#ACC0C6"],
    'PIT':["#FFB612","#101820"],
    'SEA':["#002244","#69BE28","#A5ACAF"],
    'SF' :["#AA0000","#B3995D"],
    'TB' :["#D50A0A","#FF7900","#0A0A08"],
    'TEN':["#0C2340","#4B92DB","#C8102E"],
    'WAS':["#5A1414","#FFB612"],
    'football':["#CBB67C","#663831"]
}


"""
Computes the distance between two colors in RGB space.

Params:
- hex1: The hex code of the first color.
- hex2: The hex code of the second color.

Returns:
- The color distance between the two colors.
"""
def getColorSimilarity(hex1, hex2):
    """
    Helper Method: Converts a hex color to an RGB tuple.
    """
    def get_rgb(color):
        # return np.array(tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
        return tuple(int(color[i:i+2], 16) for i in (1, 3, 5))

    if hex1 == hex2:
        return 0

    rgb1 = get_rgb(hex1)
    rgb2 = get_rgb(hex2)

    rm = 0.5 * (rgb1[0] + rgb2[0])

    r_diff = rgb1[0] - rgb2[0]
    g_diff = rgb1[1] - rgb2[1]
    b_diff = rgb1[2] - rgb2[2]

    distance = ((2 + rm / 256) * r_diff ** 2 + 4 * g_diff ** 2 +
                (3 - rm / 256) * b_diff ** 2) ** 0.5
    return distance


"""
Pairs colors for two teams. If colors are 'too close' in hue, switch to alt
color.

Params:
- team1: Name of the first team.
- team2: Name of the second team.

Returns:
- Dictionary with color assignments for both teams and football.
"""
def getColorPairs(teamA, teamB):
    teamA_colors = COLORS[teamA]
    teamB_colors = COLORS[teamB]

    # If color distance is small, switch secondary color for team2
    if getColorSimilarity(teamA_colors[0], teamB_colors[0]) < 500:
        return {
            teamA: [teamA_colors[0], teamA_colors[1]],
            teamB: [teamB_colors[1], teamB_colors[0]],
            'football': COLORS['football']
        }
    else:
        return {
            teamA: [teamA_colors[0], teamA_colors[1]],
            teamB: [teamB_colors[0], teamB_colors[1]],
            'football': COLORS['football']
        }
    
def get_x(old_y, direction):
    if direction == "left":
        return 53.3 - old_y
    else:
        return 53.3 - old_y
    
def get_y(old_x, direction):
    if direction == "left":
        return old_x
    else:
        return old_x

"""
Animates a specific play

Params:
- df: The DataFrame for the specific play

Returns:
- Visualization of the specific play
"""
def animate_play_upward(df, output_gif="play_animation.gif"):
    import os
    import plotly.graph_objects as go
    import numpy as np
    import imageio

    frames = []
    sorted_frame_list = df.frameId.unique()
    sorted_frame_list.sort()

    # Get color Combos
    team_combos = list(set(df['club'].unique()) - set(['football']))
    color_orders = getColorPairs(team_combos[0], team_combos[1])

    # General information
    direction = df['playDirection'].values[0]

    line_of_scrimmage = df['absoluteYardlineNumber'].values[0] if direction == "right" else 20 + df['absoluteYardlineNumber'].values[0]
    first_down_marker = line_of_scrimmage + df['yardsToGo'].values[0]
    play_description = df['playDescription'].values[0]

    motion_label = "None"

    # Handle long play descriptions
    if len(play_description.split(" ")) > 15 and len(play_description) > 115:
        play_description = " ".join(play_description.split(" ")[0:16]) + "<br>" + " ".join(play_description.split(" ")[16:])

    image_folder = "frames"
    os.makedirs(image_folder, exist_ok=True)

    previous_positions = {}  # To store previous positions of players
    movement_lines = []  # List to store all lines to be plotted

    for frame_id in sorted_frame_list:
        data = []

        # Add white yard lines
        for y in range(0, 120, 5):  # Draw lines every 5 yards
            data.append(
                go.Scatter(
                    x=[0, 53.3],  # Full field width
                    y=[y, y],
                    mode="lines",
                    line=dict(color="white", width=1),
                    showlegend=False,
                    hoverinfo="none",
                )
            )

        # Add line of scrimmage
        data.append(
            go.Scatter(
                y=[line_of_scrimmage, line_of_scrimmage],
                x=[0, 53.5],
                mode="lines",
                line=dict(color="blue", dash="dash"),
                showlegend=False,
                hoverinfo="none",
            )
        )

        # Add first down line
        data.append(
            go.Scatter(
                y=[first_down_marker, first_down_marker],
                x=[0, 53.5],
                mode="lines",
                line=dict(color="yellow", dash="dash"),
                showlegend=False,
                hoverinfo="none",
            )
        )

        # Add annotations for yard line numbers
        annotations = []
        for y in range(20, 110, 10):
            if y <= 60:
                text_value = str(y - 10)
            else:
                text_value = str(110 - y)
            
            annotations.append(
                dict(
                    x=2,
                    y=y,
                    text=text_value,
                    font=dict(size=18, color="white"),
                    showarrow=False,
                    textangle=90,  # Rotate text
                )
            )
            annotations.append(
                dict(
                    x=51,
                    y=y,
                    text=text_value,
                    font=dict(size=18, color="white"),
                    showarrow=False,
                    textangle=-90,  # Rotate text
                )
            )

        # Add Endzone Colors
        endzoneColors = {
            0: color_orders.get(df['homeTeamAbbr'].values[0], ["#FFFFFF"])[0],  # Default to white
            110: color_orders.get(df['visitorTeamAbbr'].values[0], ["#FFFFFF"])[0]
        }
        for x_min in [0,110]:
            data.append(
              go.Scatter(
                  y=[x_min,x_min,x_min+10,x_min+10,x_min],
                  x=[0,53.5,53.5,0,0],
                  fill="toself",
                  fillcolor=endzoneColors[x_min],
                  mode="lines",
                  line=dict(
                      color="white",
                      width=3
                  ),
                  opacity=1,
                  showlegend= False,
                  hoverinfo ="skip"
              )
            )

        # Plot Players and motion lines
        for team in df['club'].unique():
            plot_df = df.loc[(df['club'] == team) & (df['frameId'] == frame_id)].copy()

            if team != "football":
                hover_text_array = []
                for nflId in plot_df["nflId"].unique():
                    selected_player_df = plot_df.loc[plot_df["nflId"] == nflId]
                    displayName = selected_player_df["displayName"].values[0]
                    s = round(selected_player_df["s"].values[0] * 2.23693629205, 3)
                    text_to_append = f"nflId:{nflId}<br>displayName:{displayName}<br>Speed:{s} MPH"
                    hover_text_array.append(text_to_append)

                data.append(
                    go.Scatter(
                        # x=53.3 - plot_df["y"],  # Swapping x and y
                        # y=plot_df["x"],  # Swapping x and y
                        x = get_x(plot_df["y"], direction),
                        y = get_y(plot_df["x"], direction),
                        mode="markers",
                        marker=dict(
                            color=color_orders[team][0],
                            line=dict(width=3, color=color_orders[team][1]),
                            size=15,
                        ),
                        name=team,
                        hovertext=hover_text_array,
                        hoverinfo="text",
                    )
                )

                try:
                    # Add motion lines for players
                    for _, row in plot_df.iterrows():
                        player_id = row['nflId']
                        current_position = (row['x'], row['y'])

                        # Check if the player has been seen before
                        if frame_id > 1:
                            # Create a copy of the old position
                            # prev_position, _ = previous_positions[player_id]  # Access old position

                            prev_position_df = df.loc[(df['nflId'] == player_id) & (df['frameId'] == frame_id-1)].copy()
                            prev_position = (prev_position_df['x'].values[0], prev_position_df['y'].values[0])

                            # Add lines for motion and shift
                            if row['movement'] == "motion":
                                movement_lines.append({
                                    'x': [prev_position[0], current_position[0]],
                                    'y': [prev_position[1], current_position[1]],
                                    'color': 'red'
                                })

                                motion_label = row['displayName'] + ": " + row['movementClassification']

                                # Add annotation for motion classification
                                annotations.append(
                                    dict(
                                        x= 27,  # y-coordinate
                                        y= line_of_scrimmage - 8,  # x-coordinate
                                        text=motion_label,
                                        font=dict(size=20, color="white"),
                                        showarrow=False,
                                    )
                                )
                            # elif row['movement'] == "shift":                            
                            #     movement_lines.append({
                            #         'x': [prev_position[0], current_position[0]],
                            #         'y': [prev_position[1], current_position[1]],
                            #         'color': 'orange'
                            #     })
                except:
                    pass

            else:
                data.append(
                    go.Scatter(
                        x = get_x(plot_df["y"], direction),
                        y = get_y(plot_df["x"], direction),
                        mode="markers",
                        marker=dict(
                            color=color_orders[team][0],
                            line=dict(width=3, color=color_orders[team][1]),
                            size=15,
                        ),
                        name=team,
                        hoverinfo="none",
                    )
                )

        # Draw accumulated movement lines
        for line in movement_lines:
            data.append(
                go.Scatter(
                    # x=[53.3 - line["y"][0], 53.3 - line["y"][1]],
                    # y=line['x'],

                    x = [get_x(line["y"][0], direction), get_x(line["y"][1], direction)],
                    y = [get_y(line["x"][0], direction), get_y(line["x"][1], direction)],

                    mode="lines",
                    line=dict(color=line['color'], width=4),
                    showlegend=False,
                    hoverinfo="none"
                )
            )

        # Create the figure for this frame
        fig = go.Figure(data=data)
        fig.update_layout(
            autosize=False,
            width=900,
            height=900,
            xaxis=dict(range=[0, 53.3], showgrid=False, zeroline=False, visible=False),
            yaxis=dict(
                range=[line_of_scrimmage - 10, line_of_scrimmage + 20],  # Adjust y-axis range
                showgrid=False,
                zeroline=False,
                visible=False
            ),
            plot_bgcolor="#3f8c55",
            title=f"Frame {frame_id}<br>{play_description}",
            annotations=annotations,
        )


        # Save frame as an image
        frame_path = os.path.join(image_folder, f"frame_{frame_id}.png")
        fig.write_image(frame_path)
        frames.append(frame_path)

    # Pad frames to 200 if necessary
    if len(frames) < 200:
        last_frame_path = frames[-1]
        frames.extend([last_frame_path] * (200 - len(frames)))

        

    # Create a GIF
    with imageio.get_writer(output_gif, mode="I", duration=0.1) as writer:
        for frame_path in frames:
            writer.append_data(imageio.imread(frame_path))

    # Clean up frames
    for frame_path in frames:
        if os.path.exists(frame_path):
            os.remove(frame_path)

    os.rmdir(image_folder)

    print(f"Animation saved as {output_gif}")