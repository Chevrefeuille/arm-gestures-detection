from pathlib import Path
import json
from tools import skeleton as sk
import math
import matplotlib.pyplot as plt
import os


# try to smoothen data buy deleting skeleton jumps (wrong person detection)

angle_names = ['RElbow', 'RShoulder',  'LShoulder', 'LElbow']
smooth_folder = 'data/poses/smoothed/groups'

max_velocity = {}
for a in angle_names:
    max_velocity[a] = 0

pathlist = Path('data/poses/extracted/solo').glob('*.json')
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
    last_angles = sk.compute_angles(angle_names, last_coordinates)
    jumped = False
    index = 0
    for frame in ordered_frames:
        skeleton = frame['skeleton'][0]
        skeletons_json = []
        coordinates = skeleton['pose']
        score = skeleton['score']
        if len(coordinates) != 0:
            angles = sk.compute_angles(angle_names, coordinates)
            angular_velocities = sk.compute_angular_velocities(last_angles, angles)
            d = sk.average_distance(coordinates, last_coordinates)
            last_angles = angles
            last_coordinates = coordinates
            if d >= 0.015:
                jumped = not(jumped)
            if not jumped:
                skeletons_json += [{
                    'pose': coordinates,
                    'score': score
                }]
            else:
                skeletons_json += [{
                    'pose': [],
                    'score': score
                }]
        else:
            skeletons_json += [{
                'pose': [],
                'score': score
            }]
        pose_data += [{
            'skeleton': skeletons_json,
            'frame_index': index
        }]
        index += 1
            # for angle_name in sk.ANGULAR_VELOCITY_THRESHOLDS:
            #     for side in ['R', 'L']:
            #         sided_angle = side + angle_name
            #         if angular_velocities[sided_angle] > max_velocity[sided_angle]:
            #             max_velocity[sided_angle] = angular_velocities[sided_angle]

            #         if angular_velocities[sided_angle] >= sk.ANGULAR_VELOCITY_THRESHOLDS[angle_name]:
            #             print('excess')
            #         else:
            #             print('ok')
                
    smoothed_data = {}           
    smoothed_data['data'] = pose_data

    pose_file = '{}/{}.json'.format(smooth_folder, file_name)
    print(pose_file)
    with open(pose_file, 'w') as outfile:
        json.dump(smoothed_data, outfile)
                

            

