import h5py
import cv2
import numpy as np
import progressbar
import json
import operator

# prepare the video clips for annotations

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

resize_factor = 1
n_frame = 38370

with h5py.File(ground_truth_file, 'r') as f:
    data = f['trainData'].value

data = data.transpose()

with open(group_annotations_file) as f:
    group_annotations = json.load(f)

fps, width, height = 59.94, 1920, 1080 

for group_id, group_data in group_annotations.items():
    members = group_data['members']
    pedestrian_ids = list(map(int,(members)))

    # create video
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('data/clips/{}.mp4'.format(group_id),
    #     fourcc, fps, (int(width * resize_factor), int(height * resize_factor))
    # )   

    # create info frame
    info_frame = np.zeros((height, width,3), np.uint8)
    info_frame[:,:] =  (0, 0, 0)
    info_str = 'Pedestrians {}'.format(', '.join( list(map(str,(pedestrian_ids)))))
    cv2.putText(info_frame, info_str, (500, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # for i in range(100):
        # out.write(info_frame)

    int_group_data = {int(camera_id): frames for camera_id, frames in group_data['camera_frames'].items()}
    sorted_data = sorted(int_group_data.items(), key=operator.itemgetter(0))
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
            # print(first_frame, last_frame)

            video_file = 'data/videos/camera{}/0000{}.MTS'.format(camera_id, video_id)
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

                # display frame counter
                cv2.putText(frame, 'Frame {}'.format(frame_counter), (50, height - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                # display bounding boxes
                frame_data = group_data[group_data[:, 2] == frame_counter + offset]

                if frame_data.size == 0:
                    break

                for data_line in frame_data:
                    pedestrian_id = int(data_line[1])
                    # print(pedestrian_id)
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

    # out.release()

