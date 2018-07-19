import h5py
import cv2
import numpy as np
import numpy.linalg as la
import progressbar
import json
import operator
from tools import utils as ut
from tools import skeleton as sk
import matplotlib.pyplot as plt
import math


# extract all the group pose

def find_appearance(data):
    # Find the ranges of appearance for the data of a given pedestrian
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


def is_inside_roi(p, roi):
    # Check if a point is inside a given rectangle
    x1, y1 = roi[0]
    x2, y2 = roi[1]
    return p[0] >= x1 and p[0] <= x2 and p[1] >= y1 and p[1] <= y2

ground_truth_file = 'data/ground_truth/trainvalRaw.mat'
group_annotations_file = 'data/groups/groups_annotations.json'
interactions_file = 'data/groups/interactions.json'


with h5py.File(ground_truth_file, 'r') as f:
    data = f['trainData'].value
data = data.transpose()

with open(group_annotations_file) as f:
    group_annotations = json.load(f)
with open(interactions_file) as f:
    interactions = json.load(f)

fps, width, height = 59.94, 1920, 1080 


for group_id, group_data in group_annotations.items():

    pedestrian_ids = list(map(int,(group_data['members'])))
    group_interactions = interactions[group_id]

    int_group_data = {int(camera_id): frames for camera_id, frames in group_data['camera_frames'].items()}
    sorted_data = sorted(int_group_data.items(), key=operator.itemgetter(0))
    for camera_id, frames in sorted_data: 
        skeletons_sequence = {}
        for pid in pedestrian_ids:
            skeletons_sequence[pid] = []

        pose_folder = 'data/poses/camera{}'.format(camera_id)
        camera_interactions = group_interactions['camera_annotations'][str(camera_id)]
        
        # keep data for pedestrians
        group_data = data[np.logical_and(
            data[:, 0] == int(camera_id), 
            np.logical_or.reduce([data[:, 1] == i for i in pedestrian_ids])
        )]

        if not group_data.size == 0: # and group_id == '171_173':
            frame_range = find_matching_appearances(group_data, pedestrian_ids)

            first_frame = int(frame_range[0])
            last_frame = int(frame_range[1])

            group_data = group_data[np.logical_and(
                group_data[:, 2] >= first_frame,
                group_data[:, 2] <= last_frame,
            )]

            video_id = first_frame // ut.NUMBER_OF_FRAMES
            offset = ut.NUMBER_OF_FRAMES * video_id
            if camera_id == 4 and video_id > 0: # artificial offset, unknown reason
                offset += video_id * 300
                        
            first_frame = first_frame - offset
            last_frame = last_frame - offset

            for frame_counter in range(first_frame, last_frame):
                frame_data = group_data[group_data[:, 2] == frame_counter + offset]
                if frame_data.size == 0:
                    break

                # extract skeletons
                rois = {}
                for data_line in frame_data:
                    pedestrian_id = int(data_line[1])
                    x1, y1 = map(int, [data_line[3], data_line[4]])
                    x2, y2 = map(int, [x1 + data_line[5], y1 + data_line[6]])
                    x1, x2 = x1 / ut.WIDTH, x2 / ut.WIDTH
                    y1, y2 = y1 / ut.HEIGHT, y2 / ut.HEIGHT
                    rois[pedestrian_id] = [(x1, y1), (x2, y2)]
                        
                pose_file = '{}/{}_keypoints.json'.format(pose_folder, frame_counter + offset)
                with open(pose_file) as f:
                    pose_data = json.load(f)
                    skeletons = []
                    for person in pose_data['people']:
                        score, coordinates = [], []
                        keypoints = person['pose_keypoints']
                        skeleton = []
                        for i in range(0, len(keypoints), 3):
                            skeleton += [
                                [
                                    (keypoints[i], keypoints[i + 1]), 
                                    keypoints[i + 2] # score
                                ]
                            ]
                        skeletons.append(skeleton)

                skeletons = sk.find_skeletons_in_ROIs(rois, skeletons)
                # print(skeletons)
                
                for pid in pedestrian_ids:
                    skeletons_sequence[pid] += [skeletons[pid]]
            
            # print(skeletons_sequence)
            # skeletons_sequence = sk.complete_skeletons(skeletons_sequence)
            # print(skeletons_sequence)
            
            for pid in pedestrian_ids:
                json_data = {}
                pose_data = []
                for i in range(len(skeletons_sequence[pid])):
                    skeleton = skeletons_sequence[pid][i]
                    skeletons_json = []
                    coordinates, score = [], []
                    for p in skeleton:
                        coordinates += [p[0][0], p[0][1]]
                        score += [p[1]]
                    skeletons_json += [{
                        'pose': coordinates,
                        'score': score
                    }]
                    pose_data += [{
                        'skeleton': skeletons_json,
                        'frame_index': i
                    }]
        
                    
                json_data['data'] = pose_data
            
                pose_file = 'data/poses/extracted/groups/{}-{}.json'.format(pid, camera_id)
                print(pose_file)
                with open(pose_file, 'w') as outfile:
                    json.dump(json_data, outfile)
        



               

