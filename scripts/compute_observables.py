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

# compute observables using feet ground truth

data = ut.load_data()
group_annotations = ut.load_annotations()
interactions = ut.load_interactions()

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

    if len(pedestrian_ids) == 2: 
    
        group_interactions = interactions[group_id]

        intensity = group_interactions['intensity']
        
        int_group_data = {int(camera_id): frames for camera_id, frames in group_data['camera_frames'].items()}
        sorted_data = sorted(int_group_data.items(), key=operator.itemgetter(0))

        tot_coord = {}
        for pid in pedestrian_ids:
            tot_coord[pid] = []

        for camera_id, frames in sorted_data: 
            skeletons_sequence = {}
        
            camera_interactions = group_interactions['camera_annotations'][str(camera_id)]

            print(pedestrian_ids, camera_id, intensity)
            
            # keep data for pedestrians
            group_data = data[np.logical_and(
                data[:, 0] == int(camera_id), 
                np.logical_or.reduce([data[:, 1] == i for i in pedestrian_ids])
            )]

            if not group_data.size == 0:
                frame_range = ut.find_matching_appearances(group_data, pedestrian_ids)

                world_feet_coords = {}
                for pid in pedestrian_ids:
                    world_feet_coords[pid] = []

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

                    # feet_coords, frame = ut.find_feet_coordinates(frame_data, frame, draw=True)
                    world_feet_coords = ut.update_world_feet_coordinates(frame_data, world_feet_coords)

                    
                    # cv2.imshow('frame', frame)
                    # if cv2.waitKey(100) & 0xFF == ord('q'):
                    #     break

                cap.release()
                cv2.destroyAllWindows()

                for pid in pedestrian_ids:
                    tot_coord[pid] += world_feet_coords[pid]
                histograms = ut.update_observables(world_feet_coords, histograms, intensity)

                observables = ut.compute_observables(world_feet_coords)
                mean_values = ut.update_mean_observables(mean_values, observables, intensity)                


        # print(histograms)

        # for o in observables:
        #     t = [i / ut.FPS for i in range(len(observables[o]))]
        #     plt.title('{} = f(t)'.format(o))
        #     plt.xlabel('t')
        #     plt.xlabel('{}'.format(o))
        #     plt.plot(t, observables[o])
        #     plt.show()

        # for o in observables_names:
        # for pid in pedestrian_ids:
        #     coords = tot_coord[pid]
        #     # vx, vy = vx_all[pid], vy_all[pid]
        #     # t = [i / ut.FPS for i in range(len(coords))]
        #     x = [c[0] for c in coords]
        #     y = [c[1] for c in coords]
        #     plt.plot(x, y, '.', label=pid)

        # plt.title('Trajectories')
        # plt.xlabel('x')
        # plt.xlabel('y')
        # plt.legend()
        # plt.show()


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
             
               

