import scipy.io
import json
import pprint

# get group annotations from dukemtmc ground truth and create a json file

group_annotations, group_ids = {}, {}
group_id = 0
group_annotation_file = 'groups/groups_GT_def.mat'
f = scipy.io.loadmat(group_annotation_file)
groups_data = f['groups']
for line in groups_data:
    pedestrian_ids = list(map(int,line[3][0]))
    camera_id = int(line[2])
    ids_key = '_'.join(map(str, pedestrian_ids))
    if ids_key not in group_ids:
        group_ids[ids_key] = group_id
        camera_frames = {}
        camera_frames[camera_id] = list(map(int,[line[4][0][0], line[5][0][0]]))
        group_annotations[ids_key] = {
            'members': pedestrian_ids,
            'camera_frames': camera_frames
        }
        group_id += 1
    else:
        group_annotations[ids_key]['camera_frames'][camera_id] = list(map(int,[line[4][0][0], line[5][0][0]]))

with open('data/groups/groups_annotations.json', 'w') as outfile:
    json.dump(group_annotations, outfile)