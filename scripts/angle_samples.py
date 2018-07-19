import json
from tools import skeleton as sk
from tools import utils as ut
import os
import math
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# extract all the angles from the groups pose estimation

angle_names = ['RElbow', 'RShoulder', 'LElbow', 'LShoulder']

pathlist = Path('data/poses/extracted/groups').glob('*.json')
for path in pathlist:
    path_in_str = str(path)
    file_name = os.path.splitext(os.path.basename(path_in_str))[0]
    print(file_name)
    pid, cid = file_name.split('-')
    cid = int(cid)
    data = json.load(open(path_in_str))
    frames = data['data']
    ordered_frames = sorted(frames, key=lambda k: k['frame_index'])
    skeletons_data = []
    n_frames = len(frames)

    angles = {}
    for a in angle_names:
        angles[a] = []

    for frame in ordered_frames:
        skeleton = frame['skeleton'][0]
        coordinates = skeleton['pose']
        if len(coordinates) != 0:
            frame_angles = sk.compute_angles(angle_names, coordinates)
            for a in angle_names:
                angles[a] += [frame_angles[a][0] * (180 / math.pi)]
        else:
            for a in angle_names:
                angles[a] += [float('nan')]

    for a in angle_names:

        file_name =  'data/angles/samples/{}-{}-{}'.format(pid, cid, a)
        pickle.dump(angles[a], open(file_name, 'wb'))
