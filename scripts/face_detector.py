import cv2
import numpy as np
import numpy.linalg as la
import json
import operator
from tools import utils as ut
from tools import skeleton as sk
import matplotlib.pyplot as plt
import math

# try to use a Haar face detector to isolate pedestrians faces

data = ut.load_data()
group_annotations = ut.load_annotations()

group_ids = ut.compute_group_ids(group_annotations)

face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

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

                    for frame_counter in range(first_frame, last_frame):                
                        ret, frame = cap.read()

                        if (type(frame) is not np.ndarray): 
                            break
                        frame = ut.draw_info(frame, camera_id, frame_counter)

                        frame_data = camera_data[camera_data
                        [:, 2] == frame_counter + offset]
                        if frame_data.size == 0:
                            break

                        rois, frame = ut.find_rois(frame_data, frame, draw=True)

                        for (p1, p2) in rois:
                            x1, x2 = int(p1[0]*ut.WIDTH), int(p2[0]*ut.WIDTH)
                            y1, y2 = int(p1[1]*ut.HEIGHT), int(p2[1]*ut.HEIGHT)
                            
                            sub_frame = frame[y1:y2, x1:x2]
                            
                            faces = face_cascade.detectMultiScale(sub_frame, 1.3, 5)
                            for (x, y, w, h) in faces:
                                cv.rectangle(sub_frame, (x, y),(x+w, y+h), (255, 0, 0), 2)

                            cv2.imshow('frame', sub_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break                      

                        index = frame_counter - first_frame
                        skeletons = [frame['skeleton'] for frame in poses if frame['frame_index'] == index][0]
                        coordinates = skeletons[0]['pose']

                        if len(coordinates) != 0:

                            frame = sk.draw_skeletons(frame, skeletons) 
                                        
                        # out.write(frame)
                        # cv2.imshow('frame', frame)
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break

                    cap.release()
                    cv2.destroyAllWindows()



               

