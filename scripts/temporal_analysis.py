import json
from tools import skeleton as sk
from tools import utils as ut
import os
import matplotlib.pyplot as plt
import math


# extract angle and velocities from pose file

angle_names = ['RElbow', 'RShoulder']

path_in_str = 'data/poses/extracted/solo/115-2.json'
file_name = os.path.splitext(os.path.basename(path_in_str))[0]
pid, cid = file_name.split('-')
cid = int(cid)
data = json.load(open(path_in_str))
frames = data['data']
ordered_frames = sorted(frames, key=lambda k: k['frame_index'])
skeletons_data = []
n_frames = len(frames)

angles = {}
for a in angle_names:
    angles[a] = []

velocities = {}
for a in angle_names:
    velocities[a] = []

distances = []

last_coordinates = ordered_frames[0]['skeleton'][0]['pose']
last_angles = sk.compute_angles(angle_names, last_coordinates)
for frame in ordered_frames:
    skeleton = frame['skeleton'][0]
    coordinates = skeleton['pose']
    if len(coordinates) != 0:
        frame_angles = sk.compute_angles(angle_names, coordinates)
        angular_velocities = sk.compute_angular_velocities(last_angles, frame_angles)
        distances += [sk.average_stable_distance(coordinates, last_coordinates)]
        for a in angle_names:
            if angular_velocities[a] < sk.ANGULAR_VELOCITY_THRESHOLDS[a[1:]]:
                angles[a] += [frame_angles[a][0] * (180 / math.pi)]
                velocities[a] += [angular_velocities[a]]
            else:
                angles[a] += [float('nan')]
                velocities[a] += [float('nan')]
        last_angles = frame_angles
        last_coordinates = coordinates
    else:
        distances += [float('nan')]

dt = 1 / ut.FPS
t = [i * dt for i in range(n_frames)]

# plt.title('Distance between skeletons')
# plt.plot(t, distances)
# plt.xlabel('t (in s)')
# plt.ylabel('distance')
# plt.show()


for a in angle_names:
    # angles[a] = ut.median_filter(angles[a], 3)
    plt.title('{} angle'.format(a))
    plt.plot(t, angles[a])
    plt.xlabel('t (in s)')
    plt.ylabel('angle (in deg)')
    plt.show()