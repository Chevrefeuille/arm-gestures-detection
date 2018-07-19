import h5py
import cv2
import numpy as np
import progressbar

# prepare one demo sample clip


ground_truth_file = 'data/ground_truth/trainvalRaw.mat'
video_file = 'data/videos/camera1/00002.MTS'

resize_factor = 1
frame_start = 38370 + 38370

with h5py.File(ground_truth_file, 'r') as f:
    data = f['trainData'].value

# keep videos from camera 1
data = data.transpose()
data = data[data[:, 0] == 1]

cap = cv2.VideoCapture(video_file)
frames_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# keep videos from video 1
data = data[data[:, 2] <= frames_number + frame_start]

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('out.mp4',
    fourcc, fps, (int(width * resize_factor), int(height * resize_factor))
    )    

for frame_counter in progressbar.progressbar(range(frame_start, frame_start + frames_number)):
    ret, frame = cap.read()

    frame_data = data[data[:, 2] == frame_counter]
    for data_line in frame_data:
        x1, y1 = map(int, [data_line[3], data_line[4]])
        x2, y2 = map(int, [x1 + data_line[5], y1 + data_line[6]])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.rectangle(frame, (x1, y - 15), (x1 + 80, y1), (0, 255, 0), -1)
        cv2.putText(frame, str(int(data_line[1])), (x1, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
    frame = cv2.resize(frame, dsize=(0, 0), fx=resize_factor, fy=resize_factor)

    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()



