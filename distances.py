import pandas as pd
import matplotlib.pyplot as plt

def offense_distance_calculation(df):
    offense_data = df[df['isOffense']]
    line_set_frame = offense_data[offense_data['event'] == 'line_set'].frameId.min() # offensive line gets set 
    if pd.isna(line_set_frame):
            line_set_frame = offense_data.frameId.min()
    snap_frame = offense_data[offense_data['event'] == 'ball_snap'].frameId.min() # snapped ball

    presnap_data = offense_data[(offense_data['frameId'] >= line_set_frame) & 
                                (offense_data['frameId'] <= snap_frame)] # all the timepoints between set line and snapped ball

    player_distances = presnap_data.groupby('nflId').apply(lambda x: x['dis'].sum()).reset_index() #distnace by nflId, summed up over all the frames in presnap range 
    player_distances.columns = ['nflId', 'distance_traveled'] 
    print(player_distances)
    
    total_distance = player_distances['distance_traveled'].sum() # total dist 
    avg_distance = player_distances['distance_traveled'].mean() # avg dist
    
    offense_shifts = presnap_data['event'].eq('shift').sum() # find how many shift events 
    offense_motions = presnap_data['event'].eq('man_in_motion').sum() # find how many man in motion events

    return pd.Series({
        'total_offense_presnap_distance': total_distance, #yards
        'avg_offense_presnap_distance': avg_distance, #yards
        'line_set_to_snap_frames': snap_frame - line_set_frame, # frame range
        'offense_shifts': offense_shifts,
        'offense_motions': offense_motions
    })


def defense_distance_calculation(df):
    defense_data = df[df['isOffense'] == False]
    line_set_frame = defense_data[defense_data['event'] == 'line_set'].frameId.min()  # defensive line gets set 
    if pd.isna(line_set_frame):
        line_set_frame = defense_data.frameId.min()
    snap_frame = defense_data[defense_data['event'] == 'ball_snap'].frameId.min()  # snapped ball

    presnap_data = defense_data[(defense_data['frameId'] >= line_set_frame) & 
                                (defense_data['frameId'] <= snap_frame)]  # all the timepoints between set line and snapped ball

    player_distances = presnap_data.groupby('nflId').apply(lambda x: x['dis'].sum()).reset_index()  # distance by nflId, summed up over all the frames in presnap range 
    player_distances.columns = ['nflId', 'distance_traveled'] 
    print(player_distances)
    
    total_distance = player_distances['distance_traveled'].sum()  # total dist 
    avg_distance = player_distances['distance_traveled'].mean()  # avg dist
    
    shifts = presnap_data['event'].eq('shift').sum()  # find how many shift events 
    motions = presnap_data['event'].eq('man_in_motion').sum()  # find how many man in motion events

    return pd.Series({
        'total_defense_presnap_distance': total_distance,  # yards
        'avg_defense_presnap_distance': avg_distance,  # yards
        'defense_line_set_to_snap_frames': snap_frame - line_set_frame,  # frame range
        'defense_shifts': shifts,
        'defense_motions': motions
    })

def combine_res(def_res, off_res):
     return pd.merge(def_res, off_res, on='playId')