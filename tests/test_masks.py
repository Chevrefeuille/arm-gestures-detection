import h5py
import cv2
import numpy as np
import progressbar

video_file = 'videos/camera1/00002.MTS'
masks_folder = 'masks/camera1/'

resize_factor = 1
frame_start = 38370 + 38370

cap = cv2.VideoCapture(video_file)
frames_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


for frame_counter in range(frame_start, frame_start + frames_number):
    ret, frame = cap.read()

        
    frame = cv2.resize(frame, dsize=(0, 0), fx=resize_factor, fy=resize_factor)

    mask_file = '{}{}.png'.format(masks_folder, str(frame_counter))
    mask = cv2.imread(mask_file, 0)

    frame = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



