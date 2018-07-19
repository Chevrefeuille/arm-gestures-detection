import cv2
import numpy as np
import numpy.linalg as la
import numpy.fft as fft
import json
import operator
from tools import utils as ut
from tools import skeleton as sk
import matplotlib.pyplot as plt
import math
import peakutils


# perform frequency analysis to retrieve walking frequency

data = ut.load_data()
group_annotations = ut.load_annotations()

joints = ['RHip', 'LHip', 'Neck']

group_ids = ut.compute_group_ids(group_annotations)

angle_names = [] #['RElbow', 'RShoulder']

all_ids = list(map(int,set(data[:,1])))

solo_ids = [pid for pid in all_ids if pid not in group_ids]
# solo_ids = [375]

camera_ids = [1, 2, 4, 5]
# camera_ids = [5]

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

                    feet = []
                    height_deltas = {}
                    for j in joints:
                        height_deltas[j] = []
                    t = []
                    for frame_counter in range(first_frame, last_frame):                
                        ret, frame = cap.read()

                        if (type(frame) is not np.ndarray): 
                            break
                        frame = ut.draw_info(frame, camera_id, frame_counter)

                        frame_data = camera_data[camera_data
                        [:, 2] == frame_counter + offset]
                        if frame_data.size == 0:
                            break

                        feet_coord, frame = ut.find_feet_coordinates(frame_data, frame, draw=True)

                        index = frame_counter - first_frame
                        skeletons = [frame['skeleton'] for frame in poses if frame['frame_index'] == index][0]
                        coordinates = skeletons[0]['pose']

                        if len(coordinates) != 0:
                            t += [(frame_counter - first_frame) / ut.FPS]
                            frame = sk.draw_skeletons(frame, skeletons, angle_names) 

                            height_deltas = ut.update_height_deltas(height_deltas, feet_coord, coordinates)
                            feet += [feet_coord[1]]

                            last_coordinates = coordinates

                        # cv2.imshow('frame', frame)
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break

                    cap.release()
                    cv2.destroyAllWindows()

                    for j in joints:
                        xi = np.arange(len(height_deltas[j]))
                        mask = np.isfinite(height_deltas[j])
                        xfiltered = np.interp(xi, xi[mask], np.array(height_deltas[j])[mask])
                        
                        sp = fft.fft(xfiltered)
                        freq = fft.fftfreq(len(t), 1 / ut.FPS)
                        t = np.array(t)

                        # find peak
                        peaks_ind = peakutils.indexes(sp.real, thres=0, min_dist=5)
                        walk_f_ind = peaks_ind[np.argmax(sp.real[peaks_ind])]

                        walk_f = freq[walk_f_ind]

                        plt.plot(freq, sp.real)
                        plt.scatter(freq[walk_f_ind], sp.real[walk_f_ind], s=40, marker='s', color='red', label='v1')

                        # plt.plot(freq, sp2.real)
                        plt.xlabel('frequency')
                        plt.ylabel('fft')
                        plt.show()

                        # filtering data
                        order = 6
                        fs = 60
                        cutoff = walk_f / 2 

                        # y = ut.butter_lowpass_filter(xfiltered, cutoff, fs, order)

                        sintest = [0.1 + 0.01 * math.sin(2 * math.pi * i *  walk_f) for i in t]

                        plt.title('{} height'.format(j))
                        plt.plot(t, height_deltas[j])
                        plt.plot(t, sintest)
                        plt.ylim(0, 1)
                        plt.show()






               

