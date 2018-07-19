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
from tempfile import TemporaryFile

data = ut.load_data()
group_annotations = ut.load_annotations()

group_ids = ut.compute_group_ids(group_annotations)

angle_names = [] # ['RElbow', 'RShoulder']

test_group = [101, 102]
test_cid = 1

for group_id, group_data in group_annotations.items():
    pedestrian_ids = list(map(int,(group_data['members'])))
    if set(test_group) == set(pedestrian_ids):
        print(group_id)
        int_group_data = {int(camera_id): frames for camera_id, frames in group_data['camera_frames'].items()}
        sorted_data = sorted(int_group_data.items(), key=operator.itemgetter(0))
        for camera_id, frames in sorted_data:

            if camera_id == test_cid:
            
                # keep data for pedestrians
                group_data = data[np.logical_and(
                    data[:, 0] == int(camera_id), 
                    np.logical_or.reduce([data[:, 1] == i for i in pedestrian_ids])
                )]

                if not group_data.size == 0:
                    frame_range = ut.find_matching_appearances(group_data, pedestrian_ids)

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

                
                    video_file = 'data/videos/camera{}/0000{}.MTS'.format(camera_id, video_id)
                    for pid in pedestrian_ids:
                        # create video
                        rf = 0.5
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out = cv2.VideoWriter('data/clips/{}.avi'.format(pid),
                            fourcc, ut.FPS, (int(ut.WIDTH * rf), int(ut.HEIGHT * rf))
                        )   


                        pose_file = 'data/poses/extracted/groups/{}-{}.json'.format(pid, camera_id)

                        with open(pose_file) as f:
                            poses = json.load(f)['data']

                        cap = cv2.VideoCapture(video_file)
                        cap.set(1, first_frame)

                        for frame_counter in range(first_frame, last_frame):   
                            ret, frame = cap.read()

                            if (type(frame) is not np.ndarray): 
                                break

                            frame = ut.draw_info(frame, camera_id, frame_counter)

                            frame_data = group_data[group_data[:, 2] == frame_counter + offset]
                            if frame_data.size == 0:
                                break

                            rois, frame = ut.find_rois(frame_data, frame, draw=True)
                            index = frame_counter - first_frame
                            skeletons = [frame['skeleton'] for frame in poses if frame['frame_index'] == index][0]
                            frame = sk.draw_skeletons(frame, skeletons, angle_names) 

                            frame = cv2.resize(frame, dsize=(0, 0), fx=rf, fy=rf)

                            out.write(frame)
                            cv2.imshow('frame', frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                        cap.release()
                        out.release()
                        cv2.destroyAllWindows()



               

