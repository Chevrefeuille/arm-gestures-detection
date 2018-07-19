import cv2
import numpy as np
import progressbar
import json
import operator
import h5py

def find_appearance(data):
    frames = data[:,2]
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

ground_truth_file = 'ground_truth/trainvalRaw.mat'

resize_factor = 1
n_frame = 38370

pedestrian_id = 3596

with h5py.File(ground_truth_file, 'r') as f:
    data = f['trainData'].value

data = data.transpose()

fps, width, height = 59.94, 1920, 1080  

for camera_id in [1, 2, 4, 5]:

    pedestrian_data = data[np.logical_and(
        data[:, 0] == int(camera_id), 
        data[:, 1] == pedestrian_id
    )]

    if not pedestrian_data.size == 0:
        frame_ranges = find_appearance(pedestrian_data)

        for frame_range in frame_ranges:
            # find first and last frame
            first_frame = int(frame_range[0])
            last_frame = int(frame_range[1])

            all_data = data[np.logical_and(
                data[:, 0] == int(camera_id),
                np.logical_and(
                    data[:, 2] >= first_frame,
                    data[:, 2] <= last_frame,
                )
            )]

            video_id = first_frame // n_frame
            offset = n_frame * video_id
            if camera_id == 4 and video_id > 0: # artificial offset, unknown reason
                offset += 300
                        
            first_frame = first_frame - offset
            last_frame = last_frame - offset

            video_file = 'videos/camera{}/0000{}.MTS'.format(camera_id, video_id)
            # print(video_file)

            cap = cv2.VideoCapture(video_file)
            cap.set(1, first_frame)

            for frame_counter in range(first_frame, last_frame):
                ret, frame = cap.read()

                if (type(frame) is not np.ndarray): 
                    break

                # display camera id
                cv2.rectangle(frame, (10, 10), (250, 80), (255, 255, 255), -1)
                cv2.putText(frame, 'Camera {}'.format(camera_id), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                # display bounding boxes
                frame_data = all_data[all_data[:, 2] == frame_counter + offset]

                if frame_data.size == 0:
                    break

                for data_line in frame_data:
                    pedestrian_id = int(data_line[1])
                    # if pedestrian_id in members:
                    x1, y1 = map(int, [data_line[3], data_line[4]])
                    x2, y2 = map(int, [x1 + data_line[5], y1 + data_line[6]])
                    y1 -= 30 # to prevent face hidding
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                    cv2.rectangle(frame, (x1, y - 15), (x1 + 80, y1), (0, 255, 0), -1)
                    cv2.putText(frame, str(int(data_line[1])), (x1, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        
                frame = cv2.resize(frame, dsize=(0, 0), fx=resize_factor, fy=resize_factor)

                # out.write(frame)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


