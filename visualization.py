"""
Acknowledgment:
This implementation is based on the original work by Nick Wan.

Modifications have been made to adapt the code to meet specific project requirements.

Original source: https://www.kaggle.com/code/nickwan/animate-plays-with-plotly-real-no-lies-here
"""

import plotly.graph_objects as go
import numpy as np


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

"""
Animates a specific play

Params:
- df: The DataFrame for the specific play

Returns:
- Visualization of the specific play
"""
def animate_play(df):
    frames = []
    sorted_frame_list = df.frameId.unique()
    sorted_frame_list.sort()

    # Get color Combos
    team_combos = list(set(df['club'].unique())-set(['football']))
    color_orders = getColorPairs(team_combos[0],team_combos[1])

    # Get general information
    line_of_scrimmage = df['absoluteYardlineNumber'].values[0]

    # Get First Down Marker
    if df['playDirection'].values[0] == 'right':
        first_down_marker = line_of_scrimmage + df['yardsToGo'].values[0]
    else:
        first_down_marker = line_of_scrimmage - df['yardsToGo'].values[0]
    down = df['down'].values[0]
    quarter = df['quarter'].values[0]
    gameClock = df['gameClock'].values[0]
    playDescription = df['playDescription'].values[0]

    # Handle long play descriptions
    if len(playDescription.split(" "))>15 and len(playDescription)>115:
        playDescription = " ".join(playDescription.split(" ")[0:16]) + "<br>" + " ".join(playDescription.split(" ")[16:])

    updatemenus_dict = [
        {
            "buttons": [
                {
                    "args": [
                        None, {"frame": {"duration": 100, "redraw": False},
                        "fromcurrent": True, "transition": {"duration": 0}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [
                        [None], {"frame": {"duration": 0, "redraw": False},
                        "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    # Initialize plotly slider to show frame position in animation
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Frame:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    for frameId in sorted_frame_list:
        data = []

        # Add Numbers to Field
        data.append(
            go.Scatter(
                x=np.arange(20, 110, 10),
                y=[5] * len(np.arange(20, 110, 10)),
                mode='text',
                text=[str(i) for i in list(np.arange(10, 61, 10))
                      + list(np.arange(40, 9, -10))],
                textfont_size=30,
                textfont_family="Courier New, monospace",
                textfont_color="#ffffff",
                showlegend=False,
                hoverinfo='none'
            )
        )
        data.append(
            go.Scatter(
                x=np.arange(20, 110, 10),  # Generate x values once
                y=[53.5 - 5] * len(np.arange(20, 110, 10)),  # Calculate y value and repeat it based on x's length
                mode='text',
                text=[str(i) for i in list(np.arange(10, 61, 10)) + list(np.arange(40, 9, -10))],  # Simplified text generation
                textfont_size=30,
                textfont_family="Courier New, monospace",
                textfont_color="#ffffff",
                showlegend=False,
                hoverinfo='none'
            )
        )

        # Add line of scrimage
        data.append(
          go.Scatter(
              x=[line_of_scrimmage,line_of_scrimmage],
              y=[0,53.5],
              line_dash='dash',
              line_color='blue',
              showlegend=False,
              hoverinfo='none'
          )
        )

        # Add First down line
        data.append(
          go.Scatter(
              x=[first_down_marker,first_down_marker],
              y=[0,53.5],
              line_dash='dash',
              line_color='yellow',
              showlegend=False,
              hoverinfo='none'
          )
        )

        # Add Endzone Colors
        endzoneColors = {0:color_orders[df['homeTeamAbbr'].values[0]][0],
                        110:color_orders[df['visitorTeamAbbr'].values[0]][0]}
        for x_min in [0,110]:
            data.append(
              go.Scatter(
                  x=[x_min,x_min,x_min+10,x_min+10,x_min],
                  y=[0,53.5,53.5,0,0],
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

        # Plot Players
        for team in df['club'].unique():
            plot_df = df.loc[(df['club']==team)
                             & (df['frameId']==frameId)].copy()

            if team != 'football':
                hover_text_array=[]

                for nflId in plot_df['nflId'].unique():
                    selected_player_df = plot_df.loc[plot_df['nflId']==nflId]
                    nflId = int(selected_player_df['nflId'].values[0])
                    displayName = selected_player_df['displayName'].values[0]
                    s = round(selected_player_df['s'].values[0] * 2.23693629205, 3)
                    text_to_append = f"nflId:{nflId}<br>displayName:{displayName}<br>Player Speed:{s} MPH"
                    hover_text_array.append(text_to_append)

                data.append(go.Scatter(x=plot_df['x'], y=plot_df['y'],
                                mode = 'markers',
                                marker=go.scatter.Marker(
                                    color=color_orders[team][0],
                                    line=go.scatter.marker.Line(width=2,
                                    color=color_orders[team][1]),
                                    size=10
                                    ),
                                name= team, hovertext=hover_text_array, hoverinfo='text'))
            else:
                data.append(go.Scatter(x=plot_df['x'], y=plot_df['y'],
                                mode = 'markers',
                                marker=go.scatter.Marker(
                                    color=color_orders[team][0],
                                    line=go.scatter.marker.Line(
                                    width=2,
                                    color=color_orders[team][1]),
                                    size=10
                                    ),
                                name=team,hoverinfo='none'))

        # Add frame to slider
        slider_step = {'args': [
          [frameId],
          {'frame': {'duration': 100, 'redraw': False},
            'mode': 'immediate',
            'transition': {'duration': 0}}
        ],
          'label': str(frameId),
          'method': 'animate'}
        sliders_dict['steps'].append(slider_step)
        frames.append(go.Frame(data=data, name=str(frameId)))
    scale=10
    layout = go.Layout(
        autosize=False,
        width=120*scale,
        height=60*scale,
        xaxis=dict(range=[0, 120], autorange=False, tickmode='array',
                   tickvals=np.arange(10, 111, 5).tolist(),
                   showticklabels=False
                   ),
        yaxis=dict(range=[0, 53.3], autorange=False,
                   showgrid=False,showticklabels=False
                   ),

        plot_bgcolor='#00B140',

        # Create title and add play description
        title=f"GameId: {df.iloc[0].gameId},PlayId: {df.iloc[0].playId}<br>{gameClock}{quarter}Q"+"<br>"*19+f"{playDescription}",
        updatemenus=updatemenus_dict,
        sliders = [sliders_dict]
    )
    fig = go.Figure(
        data=frames[0]['data'],
        layout= layout,
        frames=frames[1:]
    )

    # Create First Down Markers
    for y_val in [0,53]:
        fig.add_annotation(
              x=first_down_marker,
              y=y_val,
              text=str(down),
              showarrow=False,
              font=dict(
                  family="Courier New, monospace",
                  size=16,
                  color="black"
                  ),
              align="center",
              bordercolor="black",
              borderwidth=2,
              borderpad=4,
              bgcolor="#ff7f0e",
              opacity=1
              )

    # Add Team Abbreviations in EndZone's
    for x_min in [0,110]:
        if x_min == 0:
            angle = 270
            teamName=df['homeTeamAbbr'].values[0]
        else:
            angle = 90
            teamName=df['visitorTeamAbbr'].values[0]
        fig.add_annotation(
          x=x_min+5,
          y=53.5/2,
          text=teamName,
          showarrow=False,
          font=dict(
              family="Courier New, monospace",
              size=32,
              color="White"
              ),
          textangle = angle
        )
    return fig