import plotly.graph_objects as go
import numpy as np

colors = {
    'ARI':"#97233F",
    'ATL':"#A71930",
    'BAL':'#241773',
    'BUF':"#00338D",
    'CAR':"#0085CA",
    'CHI':"#C83803",
    'CIN':"#FB4F14",
    'CLE':"#311D00",
    'DAL':'#003594',
    'DEN':"#FB4F14",
    'DET':"#0076B6",
    'GB':"#203731",
    'HOU':"#03202F",
    'IND':"#002C5F",
    'JAX':"#9F792C",
    'KC':"#E31837",
    'LA':"#003594",
    'LAC':"#0080C6",
    'LV':"#000000",
    'MIA':"#008E97",
    'MIN':"#4F2683",
    'NE':"#002244",
    'NO':"#D3BC8D",
    'NYG':"#0B2265",
    'NYJ':"#125740",
    'PHI':"#004C54",
    'PIT':"#FFB612",
    'SEA':"#69BE28",
    'SF':"#AA0000",
    'TB':'#D50A0A',
    'TEN':"#4B92DB",
    'WAS':"#5A1414",
    'football':'#CBB67C'
}

def animate_play(df):
    """
    Animates a specific play

    Params:
    - df: The DataFrame for the specific play

    Returns:
    - Visualization of the specific play
    """
    # General Information
    baseline = df.iloc[0]
    line_of_scrimmage = baseline['absoluteYardlineNumber']
    first_down_marker = line_of_scrimmage + baseline['yardsToGo']
    down, quarter, game_clock = baseline['down'], baseline['quarter'], baseline['gameClock']
    play_description = baseline['playDescription']

    # Handle Long Play Descriptions
    words = play_description.split(" ")
    if len(words) > 15 and len(play_description) > 115:
        play_description = " ".join(words[:16]) + "<br>" + " ".join(words[16:])

    # initialize plotly start and stop buttons for animation
    updatemenus_dict = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 100, "redraw": False},
                                "fromcurrent": True, "transition": {"duration": 0}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
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
    # initialize plotly slider to show frame position in animation
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


    frames = []

    sorted_frame_list = df.frameId.unique()
    sorted_frame_list.sort()

    for frameId in sorted_frame_list:
        data = []
        # Add Numbers to Field
        data.append(
            go.Scatter(
                x=np.arange(20,110,10),
                y=[5]*len(np.arange(20,110,10)),
                mode='text',
                text=list(map(str,list(np.arange(20, 61, 10)-10)+list(np.arange(40, 9, -10)))),
                textfont_size = 30,
                textfont_family = "Courier New, monospace",
                textfont_color = "#ffffff",
                showlegend=False,
                hoverinfo='none'
            )
        )
        data.append(
            go.Scatter(
                x=np.arange(20,110,10),
                y=[53.5-5]*len(np.arange(20,110,10)),
                mode='text',
                text=list(map(str,list(np.arange(20, 61, 10)-10)+list(np.arange(40, 9, -10)))),
                textfont_size = 30,
                textfont_family = "Courier New, monospace",
                textfont_color = "#ffffff",
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
        # Plot Players
        for team in df.club.unique():
            plot_df = df[(df.club==team)&(df.frameId==frameId)].copy()
            if team != "football":
                hover_text_array=[]
                for nflId in plot_df.nflId:
                    selected_player_df = plot_df[plot_df.nflId==nflId]
                    hover_text_array.append("nflId:{}<br>displayName:{}".format(
                        selected_player_df["nflId"].values[0],
                        selected_player_df["displayName"].values[0],
                    ))

                    ### DONT HAVE PFF POSITIONS / ROLES
                    # hover_text_array.append("nflId:{}<br>displayName:{}<br>Position:{}<br>Role:{}".format(
                        # selected_player_df["nflId"].values[0],
                        # selected_player_df["displayName"].values[0],
                        # selected_player_df["pff_positionLinedUp"].values[0],
                        # selected_player_df["pff_role"].values[0]
                    # ))

                data.append(go.Scatter(x=plot_df["x"], y=plot_df["y"],mode = 'markers',marker_color=colors[team],name=team,hovertext=hover_text_array,hoverinfo="text"))
            else:
                data.append(go.Scatter(x=plot_df["x"], y=plot_df["y"],mode = 'markers',marker_color=colors[team],name=team,hoverinfo='none'))
        # add frame to slider
        slider_step = {"args": [
            [frameId],
            {"frame": {"duration": 100, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": 0}}
        ],
            "label": str(frameId),
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)
        frames.append(go.Frame(data=data, name=str(frameId)))

    scale=10
    layout = go.Layout(
        autosize=False,
        width=120*scale,
        height=60*scale,
        xaxis=dict(range=[0, 120], autorange=False, tickmode='array',tickvals=np.arange(10, 111, 5).tolist(),showticklabels=False),
        yaxis=dict(range=[0, 53.3], autorange=False,showgrid=False,showticklabels=False),

        plot_bgcolor='#00B140',
        # Create title and add play description at the bottom of the chart for better visual appeal
        title=f"GameId: {df.gameId.unique()[0]}, PlayId: {df.playId.unique()[0]}<br>{game_clock} {quarter}Q"+"<br>"*19+f"{play_description}",
        updatemenus=updatemenus_dict,
        sliders = [sliders_dict]
    )

    fig = go.Figure(
        data=frames[0]["data"],
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

    return fig