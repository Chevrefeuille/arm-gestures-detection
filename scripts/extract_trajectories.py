import cv2
import numpy as np
import numpy.linalg as la
import json
import operator
from tools import utils as ut
from tools import skeleton as sk
import matplotlib.pyplot as plt
import math

# extract world trajectories from openpose estimations

data = ut.load_data()
group_annotations = ut.load_annotations()

group_ids = ut.compute_group_ids(group_annotations)

angle_names = [] #['RElbow', 'RShoulder']

all_ids = list(map(int,set(data[:,1])))

solo_ids = [pid for pid in all_ids if pid not in group_ids]
# solo_ids = [136]
# 
camera_ids = [1, 2, 4, 5]
# camera_ids = [1]


for pedestrian_id in solo_ids:
    pedestrian_data = data[data[:, 1] == pedestrian_id]
    if not pedestrian_data.size == 0:
        appearances = ut.find_appearances(pedestrian_data)
        for camera_id, frame_ranges in appearances.items():
            if camera_id in camera_ids:                
                # keep data for pedestrian
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
                video_id = first_frame // ut.NUMBER_OF_FRAMES
                if video_id >= 2 and video_id <= 3:
                    offset = ut.NUMBER_OF_FRAMES * video_id
                    if camera_id == 4 and video_id > 0: # artificial offset, unknown reason
                        offset += video_id * 300
                                
                    first_frame = first_frame - offset
                    last_frame = last_frame - offset
                    # print(first_frame, last_frame)

                    video_file = 'data/videos/camera{}/0000{}.MTS'.format(camera_id, video_id)
                    pose_file = 'data/poses/extracted/solo/{}-{}.json'.format(pedestrian_id, camera_id)
                    print(pose_file)

                    with open(pose_file) as f:
                        poses = json.load(f)['data']

                    cap = cv2.VideoCapture(video_file)
                    cap.set(1, first_frame)

                    rvec = ut.CAMERA_RVECS[camera_id]
                    tvec = ut.CAMERA_TVECS[camera_id] 
                    inv_rvec = np.linalg.inv(rvec)
                    
                    x, y = [], []
                    x2, y2 = [], []

                    for frame_counter in range(first_frame, last_frame):                
                        ret, frame = cap.read()

                        if (type(frame) is not np.ndarray): 
                            break
                        frame = ut.draw_info(frame, camera_id, frame_counter)

                        frame_data = camera_data[camera_data[:, 2] == frame_counter + offset]
                        if frame_data.size == 0:
                            break

                        rois, frame = ut.find_rois(frame_data, frame, draw=True)

                        index = frame_counter - first_frame
                        skeletons = [frame['skeleton'] for frame in poses if frame['frame_index'] == index][0]

                        i_neck = sk.JOINTS['Neck']
                        neck_x, neck_y = skeletons[0]['pose'][2 * i_neck], skeletons[0]['pose'][2 * i_neck+1]

                        test_x, test_y = frame_data[0][9], frame_data[0][10]
                        
                        if test_x != 0 and test_y != 0:

                            homogene_neck_coordinates = np.array([
                                test_x, test_y, 1
                            ])

                            dvec = np.dot(inv_rvec, homogene_neck_coordinates)
                            
                            neck_world_x = (tvec[2] / dvec[2]) * dvec[0] - tvec[0]
                            neck_world_y = (tvec[2] / dvec[2]) * dvec[1] - tvec[1]

                            x += [neck_world_x]
                            # x2 += [frame_data[0][7]]
                            y += [neck_world_y]
                            # y2 += [frame_data[0][8]]

                        if neck_x != 0 and neck_y != 0:

                            homogene_neck_coordinates2 = np.array([
                                neck_x, neck_y, 1
                            ])
                            dvec2 = np.dot(inv_rvec, homogene_neck_coordinates2)
                            
                            neck_world_x2 = (tvec[2] / dvec2[2]) * dvec2[0] - tvec[0]
                            neck_world_y2 = (tvec[2] / dvec2[2]) * dvec2[1] - tvec[1]

                            x2 += [neck_world_x2]
                            y2 += [neck_world_y2]
                    
                    t = range(len(x))
                    plt.plot(x, y, label='comp')
                    t2 = range(len(x2))
                    plt.plot(x2, y2)
                    plt.legend()
                    plt.show()

                    cap.release()
                    cv2.destroyAllWindows()



               

