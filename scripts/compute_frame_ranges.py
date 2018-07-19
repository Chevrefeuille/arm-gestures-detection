import h5py
import cv2
import numpy as np
import numpy.linalg as la
import progressbar
import json
import operator
from tools import utils as ut

# compute the frame ranges of each group

def find_appearance(data):
    # Find the ranges of appearance for the data a given pedestrian
    frames = data[:,2]
    if frames.size != 0:
        break_points = []
        prev_frame = frames[0]
        for i in range(1, len(frames)):
            if frames[i] != prev_frame + 1:
                break_points += [i]
            prev_frame = frames[i]
        frame_ranges = []
        first_frame = frames[0]
        last_frame = frames[0]
        for b in break_points:
            last_frame = frames[b - 1]
            frame_ranges += [(first_frame, last_frame)] 
            first_frame =  frames[b]
        frame_ranges += [(first_frame, frames[-1])] 
        return frame_ranges
    return [(0, 0)]


def find_matching_appearances(group_data, pedestrian_ids):
    # Find the range of appearance of a group of pedestrian
    ranges = {}
    for pedestrian_id in pedestrian_ids:
        pedestrian_data = group_data[group_data[:, 1] == pedestrian_id]
        frame_ranges = find_appearance(pedestrian_data)
        ranges[pedestrian_id] = frame_ranges
    ranges_1 = ranges[pedestrian_ids[0]]
    ranges_2 = ranges[pedestrian_ids[1]]
    fit_frames, max_fit = (), 0
    for r1 in ranges_1:
        for r2 in ranges_2:
            fit = min(r1[1], r2[1]) - max(r1[0], r2[0])
            if fit > max_fit:
                max_fit = fit
                fit_frames = (max(r1[0], r2[0]), min(r1[1], r2[1]))
    tot_fit_frames, tot_max_fit = (), 0
    if len(pedestrian_ids) == 3:
        ranges_3 = ranges[pedestrian_ids[2]]
        for r3 in ranges_3:
            fit = min(fit_frames[1], r3[1]) - max(fit_frames[0], r3[0])
            if fit > tot_max_fit:
                tot_max_fit = fit
                tot_fit_frames = (max(fit_frames[0], r3[0]), min(fit_frames[1], r3[1]))
        return tot_fit_frames
    return fit_frames  

ground_truth_file = 'data/ground_truth/trainvalRaw.mat'
group_annotations_file = 'data/groups/groups_annotations.json'

group_info_file = 'data/groups/groups_info.json'

n_frame = 38370

with h5py.File(ground_truth_file, 'r') as f:
    data = f['trainData'].value

data = data.transpose()

with open(group_annotations_file) as f:
    group_annotations = json.load(f)

groups_info = {}

for group_id, group_data in group_annotations.items():
    members = group_data['members']
    pedestrian_ids = list(map(int,(members)))

    int_group_data = {int(camera_id): frames for camera_id, frames in group_data['camera_frames'].items()}
    sorted_data = sorted(int_group_data.items(), key=operator.itemgetter(0))

    group_info = {}

    for camera_id, frames in sorted_data: 
        
        # keep data for pedestrians
        group_data = data[np.logical_and(
            data[:, 0] == int(camera_id), 
            np.logical_or.reduce([data[:, 1] == i for i in pedestrian_ids])
        )]

        if not group_data.size == 0:
            frame_range = find_matching_appearances(group_data, pedestrian_ids)

            first_frame = int(frame_range[0])
            last_frame = int(frame_range[1])

            group_data = group_data[np.logical_and(
                group_data[:, 2] >= first_frame,
                group_data[:, 2] <= last_frame,
            )]

            video_id = first_frame // n_frame
            offset = n_frame * video_id
            if camera_id == 4 and video_id > 0: # artificial offset, unknown reason
                offset += 300
                        
            first_frame = first_frame - offset
            last_frame = last_frame - offset

            camera_info = {
                'video_id': video_id,
                'first_frame': first_frame,
                'last_frame': last_frame
            }

            group_info[camera_id] = camera_info

    groups_info[group_id] = group_info

with open(group_info_file, 'w') as outfile:
    json.dump(groups_info, outfile)



            
