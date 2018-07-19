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

# compute the observables using OpenPose neck position

data = ut.load_data()
group_annotations = ut.load_annotations()
interactions = ut.load_interactions()


group_ids = ut.compute_group_ids(group_annotations)

angle_names = [] # ['RElbow', 'RShoulder']

intensities = ['0', '1', '2', '3']
observables_names = ['d', 'v_G', 'v_diff']

histograms = {}
mean_values = {}
for i in intensities:
    histograms[i] = {}
    mean_values[i] = {}
    for o in observables_names:
        histograms[i][o] = np.array([])
        mean_values[i][o] = np.array([])

for group_id, group_data in group_annotations.items():
    pedestrian_ids = list(map(int,(group_data['members'])))
    print(pedestrian_ids)
    
    if len(pedestrian_ids) == 2: 
        group_interactions = interactions[group_id]
        intensity = group_interactions['intensity'] 

        int_group_data = {int(camera_id): frames for camera_id, frames in group_data['camera_frames'].items()}
        sorted_data = sorted(int_group_data.items(), key=operator.itemgetter(0))
        for camera_id, frames in sorted_data:

            # if camera_id == test_cid:
            
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
                neck_coordinates = {}
                feet_coordinates = {}
                
                for pid in pedestrian_ids:
                    # create video
                    rf = 0.5
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter('data/clips/{}.avi'.format(pid),
                        fourcc, ut.FPS, (int(ut.WIDTH * rf), int(ut.HEIGHT * rf))
                    )   

                    neck_coordinates[pid] = []
                    feet_coordinates[pid] = []

                    pose_file = 'data/poses/extracted/groups/{}-{}.json'.format(pid, camera_id)

                    with open(pose_file) as f:
                        poses = json.load(f)['data']

                    cap = cv2.VideoCapture(video_file)
                    cap.set(1, first_frame)

                    rvec = ut.CAMERA_RVECS[camera_id]
                    tvec = ut.CAMERA_TVECS[camera_id] 
                    inv_rvec = np.linalg.inv(rvec)
                    i_neck = sk.JOINTS['Neck']

                    for frame_counter in range(first_frame, last_frame):   
                        ret, frame = cap.read()

                        if (type(frame) is not np.ndarray): 
                            break

                        frame = ut.draw_info(frame, camera_id, frame_counter)

                        frame_data = group_data[group_data[:, 2] == frame_counter + offset]
                        if frame_data.size == 0:
                            break

                        rois, frame = ut.find_rois(frame_data, frame, draw=False)
                        index = frame_counter - first_frame
                        skeletons = [frame['skeleton'] for frame in poses if frame['frame_index'] == index][0]


                        if len(skeletons[0]['pose']) > 0:
                            neck_x, neck_y = skeletons[0]['pose'][2 * i_neck], skeletons[0]['pose'][2 * i_neck+1]       
                        else:
                            neck_x, neck_y = 0, 0          

                        homogene_neck_coordinates = np.array([
                            neck_x, neck_y, 1
                        ])
                        dvec = np.dot(inv_rvec, homogene_neck_coordinates)
                        
                        neck_world_x = (tvec[2] / dvec[2]) * dvec[0] - tvec[0]
                        neck_world_y = (tvec[2] / dvec[2]) * dvec[1] - tvec[1]

                        neck_coordinates[pid] += [(neck_world_x, neck_world_y)]
                        feet_coordinates[pid] += [(frame_data[0][7], frame_data[0][8])]

                        # out.write(frame)
                        # cv2.imshow('frame', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()

                histograms = ut.update_observables(neck_coordinates, histograms, intensity)
                observables = ut.compute_observables(neck_coordinates)
                mean_values = ut.update_mean_observables(mean_values, observables, intensity)

                for pid in pedestrian_ids:
                    coords = neck_coordinates[pid]
                    coords_2 = feet_coordinates[pid]
                    # vx, vy = vx_all[pid], vy_all[pid]
                    # t = [i / ut.FPS for i in range(len(coords))]
                    x = [c[0] for c in coords]
                    tx = range(len(x))
                    y = [c[1] for c in coords]
                    ty = range(len(y))
                    x2 = [c[0] for c in coords_2]
                    y2 = [c[1] for c in coords_2]
                    plt.plot(tx, x, '.', label='neck' + str(pid))
                    plt.plot(tx, x2, '.', label='feet '+ str(pid))
                plt.legend()
                plt.show()    

               
for i in intensities:
    for o in observables_names:
        histograms[i][o] = histograms[i][o].tolist()

json_file = 'data/observables/obs.json'
with open(json_file, 'w') as outfile:
    json.dump(histograms, outfile)   

for i in intensities:
    for o in observables_names:
        mean_values[i][o] = mean_values[i][o].tolist()

json_file = 'data/observables/mean_obs.json'
with open(json_file, 'w') as outfile:
    json.dump(mean_values, outfile)



            


