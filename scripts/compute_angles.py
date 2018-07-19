from pathlib import Path
import json
from tools import skeleton as sk
import math
import matplotlib.pyplot as plt

# compute angles for groups and solo pedestrians

angle_names = ['RElbow', 'RShoulder',  'LShoulder', 'LElbow']
angles = {}

group_annotations_file = 'data/groups/groups_annotations.json'
with open(group_annotations_file) as f:
    group_annotations = json.load(f)

group_ids = group_annotations.keys()

interactions_file = 'data/groups/interactions.json'
with open(interactions_file) as f:
    interactions = json.load(f)

for camera_id in [1, 2, 4, 5]:
    angles[camera_id] = {}
    for gesture in ['0', '1']:
        angles[camera_id][gesture] = {}
        # init angle tables
        for a in angle_names:
            angles[camera_id][gesture][a] = []
        pathlist = Path('data/poses/extracted/groups').glob('*.json')
        for path in pathlist:
            path_in_str = str(path)
            pid, cid = path_in_str.split('/')[-1].split('.')[0].split('-')
            cid = int(cid)
            gids = [gid for gid in group_ids if pid in gid.split('_')]
            gid = [gid for gid in gids if str(cid) in interactions[gid]['camera_annotations']][0]
            if cid == camera_id and interactions[gid]['camera_annotations'][str(cid)][2] == gesture:
                data = json.load(open(path_in_str))
                frames = data['data']
                angles = sk.update_histograms(frames, angles, angle_names, camera_id, gesture)
        pathlist = Path('data/poses/extracted/solo').glob('*.json')
    for path in pathlist:
        path_in_str = str(path)
        pid, cid = path_in_str.split('/')[-1].split('.')[0].split('-')
        cid = int(cid)
        if cid == camera_id:
            data = json.load(open(path_in_str))
            frames = data['data']
            angles = sk.update_histograms(frames, angles, angle_names, camera_id,  '0')

    for angle_name in angle_names:
        file_1 = 'data/angles/{}_{}_{}.json'.format(camera_id, angle_name, '0')
        file_2 = 'data/angles/{}_{}_{}.json'.format(camera_id, angle_name, '1')
        with open(file_1, 'w') as outfile:
            json.dump(angles[camera_id]['0'][angle_name], outfile)
        with open(file_2, 'w') as outfile:
            json.dump(angles[camera_id]['1'][angle_name], outfile)


    
    
