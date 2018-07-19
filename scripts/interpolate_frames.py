from pathlib import Path
import json
from tools import skeleton as sk
import math
import matplotlib.pyplot as plt
import os

# try to interpolate missing estimations

angle_names = ['RElbow', 'RShoulder',  'LShoulder', 'LElbow']
interp_folder = 'data/poses/interpolated/groups'

max_velocity = {}
for a in angle_names:
    max_velocity[a] = 0

base_score = [0.5 for i in range(18)]

pathlist = Path('data/poses/smoothed/groups').glob('*.json')
for path in pathlist:
    path_in_str = str(path)
    file_name = os.path.splitext(os.path.basename(path_in_str))[0]
    pid, cid = file_name.split('-')
    cid = int(cid)
    data = json.load(open(path_in_str))
    frames = data['data']
    ordered_frames = sorted(frames, key=lambda k: k['frame_index'])
    skeletons_data = []
    n_frames = len(frames)
    pose_data = []
    last_coordinates = ordered_frames[0]['skeleton'][0]['pose']
    gap = False
    index = 0
    for frame in ordered_frames:
        skeleton = frame['skeleton'][0]
        coordinates = skeleton['pose']
        score = skeleton['score']
        if not gap:
            if len(coordinates) == 0:
                start_gap_coordinates = last_coordinates
                start_gap_index = index - 1 
                gap = True
            else:
                pose_data += [{
                    'skeleton': [{
                        'pose': coordinates,
                        'score': score
                    }],
                    'frame_index': index
                }]
                last_coordinates = coordinates
        else:
            if len(coordinates) != 0:
                end_gap_coordinates = coordinates
                end_gap_index = index
                pose_data += [{
                    'skeleton': [{
                        'pose': coordinates,
                        'score': score
                    }],
                    'frame_index': index
                }]
                missing_coordinates = sk.interpolate(
                    start_gap_coordinates, end_gap_coordinates, 
                    start_gap_index, end_gap_index
                )
                for int_index, coordinates in missing_coordinates:
                    pose_data += [{
                        'skeleton': [{
                            'pose': coordinates,
                            'score': base_score
                        }],
                        'frame_index': int_index
                    }]
                last_coordinates = coordinates
                gap = False

        index += 1
                
    smoothed_data = {}           
    smoothed_data['data'] = pose_data

    pose_file = '{}/{}.json'.format(smooth_folder, file_name)
    # print(pose_file)
    with open(pose_file, 'w') as outfile:
        json.dump(smoothed_data, outfile)
                

            

