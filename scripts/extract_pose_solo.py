import h5py
import cv2
import numpy as np
import progressbar
import json
import operator
from  tools import utils as ut
from tools import skeleton as sk

# extract all the solo pose

def find_appearances(data):
    appearances = {}
    for camera_id in [1, 2, 4, 5]:
        camera_data = data[data[:, 0] == int(camera_id)]
        if camera_data.size != 0:
            frames = camera_data[:,2]
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
                appearances[camera_id] = frame_ranges
    return appearances


ground_truth_file = 'data/ground_truth/trainvalRaw.mat'
group_annotations_file = 'data/groups/groups_annotations.json'

dyad_ids = []

with open(group_annotations_file) as f:
    group_annotations = json.load(f)

for group_id, group_data in group_annotations.items():
    members = group_data['members']
    pedestrian_ids = list(map(int,(members)))
    for pid in pedestrian_ids:
        if pid not in dyad_ids:
            dyad_ids += [pid]

resize_factor = 1
n_frame = 38370

with h5py.File(ground_truth_file, 'r') as f:
    data = f['trainData'].value

data = data.transpose()
all_ids = list(map(int,set(data[:,1])))

solo_ids = [pid for pid in all_ids if pid not in dyad_ids]

with open(group_annotations_file) as f:
    group_annotations = json.load(f)

fps, width, height = 59.94, 1920, 1080 

camera_ids = [1, 2, 4, 5]
for pedestrian_id in solo_ids:
    pedestrian_data = data[data[:, 1] == pedestrian_id]
    if not pedestrian_data.size == 0:
        appearances = find_appearances(pedestrian_data)
        for camera_id, frame_ranges in appearances.items():
            if camera_id in camera_ids:
                skeletons_sequence = []
                pose_folder = 'data/poses/camera{}'.format(camera_id)
                frame_range = frame_ranges[0]
                first_frame = int(frame_range[0])
                last_frame = int(frame_range[1])

                camera_data = pedestrian_data[np.logical_and(
                    pedestrian_data[:, 0] == camera_id,
                    np.logical_and(
                        pedestrian_data[:, 2] >= first_frame,
                        pedestrian_data[:, 2] <= last_frame
                    )
                )]

                video_id = first_frame // n_frame
                if video_id >= 2 and video_id <= 3:
                    offset = n_frame * video_id
                    if camera_id == 4 and video_id > 0: # artificial offset, unknown reason
                        offset += video_id * 300
                                
                    first_frame = first_frame - offset
                    last_frame = last_frame - offset
                    # print(first_frame, last_frame)

                    video_file = 'data/videos/camera{}/0000{}.MTS'.format(camera_id, video_id)
                  
                    for frame_counter in range(first_frame, last_frame):
                        frame_data = camera_data[camera_data[:, 2] == frame_counter + offset]
                        # print(frame_data)
                        if frame_data.size == 0:
                            break

                        # extract skeletons
                        rois = []
                        for data_line in frame_data:
                            pedestrian_id = int(data_line[1])
                            x1, y1 = map(int, [data_line[3], data_line[4]])
                            x2, y2 = map(int, [x1 + data_line[5], y1 + data_line[6]])
                            x1, x2 = x1 / ut.WIDTH, x2 / ut.WIDTH
                            y1, y2 = y1 / ut.HEIGHT, y2 / ut.HEIGHT
                            rois.append([(x1, y1), (x2, y2)])
                                
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

                        skeletons_sequence += [skeletons]
                    
                    # print(skeletons_sequence)
                    # skeletons_sequence = sk.complete_skeletons(skeletons_sequence)
                    # print(skeletons_sequence)
                    
                    json_data = {}
                    pose_data = []
                    for i in range(len(skeletons_sequence)):
                        skeletons = skeletons_sequence[i]
                        skeletons_json = []
                        for skeleton in skeletons:
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
                
                    pose_file = 'data/poses/extracted/solo/{}-{}.json'.format(pedestrian_id, camera_id)
                    print(pose_file)
                    with open(pose_file, 'w') as outfile:
                        json.dump(json_data, outfile)
                                
