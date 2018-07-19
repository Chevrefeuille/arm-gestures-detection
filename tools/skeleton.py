import itertools
import matplotlib.pyplot as plt
import json
import matplotlib.animation as animation
import cv2
from tools import utils as ut
import math

JOINTS = {
    'Nose': 0,
    'Neck': 1,
    'RShoulder': 2,
    'RElbow': 3,
    'RWrist': 4,
    'LShoulder': 5, 
    'LElbow': 6,
    'LWrist': 7,
    'RHip': 8,
    'RKnee': 9,
    'RAnkle': 10,
    'LHip': 11,
    'LKnee': 12,
    'LAnkle': 13,
    'REye': 14,
    'LEye': 15,
    'REar': 16,
    'LEar': 17,
    'Background': 18
    }

LIMBS = {
    'RForearm': {
        'value': 1,
        'joints': [4, 3]
    },
    'RUpperArm': {
        'value': 2,
        'joints': [2, 3]
    },
    'RTrapeze' : {
        'value': 3,
        'joints': [2, 1]
    },
    'LForearm': {
        'value': 4,
        'joints': [7, 6]
    },
    'LUpperArm': {
        'value': 5,
        'joints': [5, 6]
    },
    'LTrapeze': {
        'value': 6,
        'joints': [5, 1]
    },
    'RTorso': {
        'value': 7,
        'joints': [8, 1]
    },
    'LTorso': {
        'value': 8,
        'joints': [11, 1]
    },
    'LUpperLeg': {
        'value': 9,
        'joints': [12, 11]
    },
    'RUpperLeg': {
        'value': 10,
        'joints': [9, 8]
    },
    'LLowerLeg': {
        'value': 11,
        'joints': [12, 13]
    },
    'RLowerLeg': {
        'value': 12,
        'joints': [9, 10]
    }
}

ANGLES = {
    'RShoulder': {
        'value': 1,
        'limbs': [2, 3],
        'position': 2
    },
    'LShoulder': {
        'value': 2,
        'limbs': [5, 6],
        'position': 5
    },
    'RElbow': {
        'value': 3,
        'limbs': [1, 2],
        'position': 3
    },
    'LElbow': {
        'value': 4,
        'limbs': [4, 5],
        'position': 6
    },

}

ANGULAR_VELOCITY_THRESHOLDS = { # rad/s
    'Elbow': 25,
    'Shoulder':18
}

JUMP_THRESHOLD = 0.03


STABLE_JOINTS = ['Neck', 'RHip', 'LHip']

def find_most_confident(skeletons):
    """
    Return the two skeletons with the largest average confidences and closest gravity center
    """
    data = []
    for skeleton in skeletons:
        n_joints = len(skeleton['score'])
        coordinates = skeleton['pose']
        average_confidence = sum(skeleton['score']) / n_joints
        x, y = [], []
        for i in range(0, len(coordinates), 2):
            x += [coordinates[i]]
            y += [coordinates[i+1]]
        center_of_gravity = (sum(x) / n_joints, sum(y) / n_joints)
        data += [(skeleton, average_confidence, center_of_gravity)]
    sorted_by_confidence = sorted(data, key=lambda c: c[1], reverse=True)
    best_s1, best_s2 = sorted_by_confidence[0][0], sorted_by_confidence[0][0]
    best_distance = 1
    for pair in itertools.combinations(sorted_by_confidence, 2):
        s1, _, g1 = pair[0]
        s2, _, g2 = pair[1]
        distance = ((g1[0] - g2[0])**2 + (g1[1] - g2[1])**2)**.5
        if distance < best_distance:
            best_s1, best_s2 = s1, s2
    
    return best_s1, best_s2

def find_consistent(skeletons, s1):
    """
    Return the skeleton which is consistent with the skeletons s1
    """
    average_distances = []
    for skeleton in skeletons:
        coordinates = skeleton['pose']
        average_distance = 0
        n_joints = len(skeleton['score'])
        for i in range(0, len(coordinates), 2):
            x1, y1 = s1['pose'][i], s1['pose'][i+1]
            x, y = coordinates[i], coordinates[i+1]
            d = ((x1 - x)**2 + (y1 - y)**2)**.5
            average_distance += d
        average_distance /= n_joints
        # print(average_distance)
        average_distances += [(skeleton, average_distance)]

    sorted_by_distance = sorted(average_distances, key=lambda c: c[1])
    return sorted_by_distance[0][0]


def complete_skeletons(sequence):
    """
    Complete missing joints of a sequence of skeletons with consistent guessed values
    """
    previous_values = sequence[0]
    for t in range(len(sequence)):
        for n in range(len(sequence[t])):
            for i in range(len(sequence[t][n])):
                coordinates = sequence[t][n][i][0]
                if coordinates != (0, 0):
                    previous_values[n][i] = sequence[t][n][i]
                else:
                    # print(previous_values[i])
                    sequence[t][n][i] = previous_values[n][i]
    return sequence

def update_plot(i, data, scat):
    scat.set_offsets(data[i])
    return scat,

def plot_skeletons(file_path):
    data = json.load(open(file_path))
    frames = data['data']
    ordered_frames = sorted(frames, key=lambda k: k['frame_index'])
    skeletons_data = []
    n_frames = len(frames)
    for frame in ordered_frames:
        # print(frame['frame_index'])
        frame_skeletons = []
        skeletons = frame['skeleton']
        for skeleton in skeletons:
            coordinates = skeleton['pose']
            for i in range(0, len(coordinates), 2):
                frame_skeletons += [[coordinates[i], 1 - coordinates[i+1]]]
        skeletons_data.append(frame_skeletons)
    
    # display the skeletons animation
    fig = plt.figure()
    scat = plt.scatter(skeletons_data[0][0], skeletons_data[0][1])
    ani = animation.FuncAnimation(fig, update_plot, frames=range(n_frames),
                                    fargs=(skeletons_data, scat),interval=10, repeat=False)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.show()


def draw_skeletons(frame, skeletons, angle_names=[]):
    # Draw the skeletons
    for skeleton in skeletons:
        coordinates = skeleton['pose']
        scores = skeleton['score']
        if len(coordinates) != 0:
            for i in range(1, 14):
                score = scores[i]
                if score <= 0.2:
                    p = (0, 0)
                p = (coordinates[2 * i], coordinates[2 * i + 1])
                p = (int(p[0] * ut.WIDTH), int(p[1] * ut.HEIGHT))
                if p != (0, 0):
                    cv2.circle(frame, p, int(5 * scores[i]), (0, 255, 255), -1)
            for _, l_data in LIMBS.items():
                joints = l_data['joints']
                p1 = (coordinates[2 * joints[0]], coordinates[2 * joints[0] + 1])
                p2 = (coordinates[2 * joints[1]], coordinates[2 * joints[1] + 1])
                p1 = (int(p1[0] * ut.WIDTH), int(p1[1] * ut.HEIGHT))
                p2 = (int(p2[0] * ut.WIDTH), int(p2[1] * ut.HEIGHT))
                if p1 != (0, 0) and p2 != (0, 0):
                    cv2.line(frame, p1, p2, (255, 0, 0), 2)
            frame_angles = compute_angles(angle_names, coordinates)
            for angle_name, (angle_value, angle_pos) in frame_angles.items():
                if not math.isnan(angle_value):
                    angle = round(angle_value  * (180 / math.pi), 1)
                    pos = (int(angle_pos[0] * ut.WIDTH), int(angle_pos[1] * ut.HEIGHT))
                    cv2.putText(frame, str(angle), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame

def is_inside_roi(p, roi):
    # Check if a point is inside a given rectangle
    x1, y1 = roi[0]
    x2, y2 = roi[1]
    return p[0] >= x1 and p[0] <= x2 and p[1] >= y1 and p[1] <= y2

def find_skeletons_in_ROIs(rois, skeletons):
    best_skeletons = {}
    for pid, roi in rois.items():
        best_skeleton, best_score = [], 0
        for skeleton in skeletons:
            score, i = 0, 0
            for p, _ in skeleton:
                if p[0] != 0 and p[1] != 0:
                    if is_inside_roi(p, roi):
                        score += 1
                    i += 1
            score /= i
            if score >= best_score:
                best_score = score
                best_skeleton = skeleton
        best_skeletons[pid] = best_skeleton
    return best_skeletons

def compute_angles(angle_names, skeleton):
    """
    Compute the angles given the skeleton coordinates
    """
    angles = {}
    for a in angle_names:
        limbs = ANGLES[a]['limbs']
        vectors = []
        for l in limbs:
            lname = [name for name in LIMBS if LIMBS[name]['value'] == l][0]
            joints = LIMBS[lname]['joints']
            p1 = (skeleton[2 * joints[0]], skeleton[2 * joints[0] + 1])
            p2 = (skeleton[2 * joints[1]], skeleton[2 * joints[1] + 1])
            if p1 != (0, 0) and p2 != (0, 0):
                v = [
                    p2[0] - p1[0],
                    p2[1] - p1[1]
                ]
                # plt.plot([skeleton[2 * joints[0]], skeleton[2 * joints[1]]], [1 - skeleton[2 * joints[0] + 1], 1 - skeleton[2 * joints[1] + 1]], marker = 'o')
                vectors += [v]
            else:
                vectors += [[0, 0]]
        # plt.xlim((0, 1))
        # plt.ylim((0, 1))
        # plt.show()
        position_index = ANGLES[a]['position']
        position = (skeleton[2 * position_index], skeleton[2 * position_index + 1])
        angles[a] = (ut.compute_angle(vectors[0], vectors[1]), position)
        # print(angles[a])
    return angles


def average_distance(c1, c2):
    """
    Compute average euclidian distance (joint to joint) between two skeletons
    """
    distances = []
    for i in range(0, len(c1), 2):
        if c1[i] != 0 and c1[i+1] != 0 and c2[i] != 0 and c2[i+1] != 0:
            distances += [((c2[i] - c1[i])**2 + (c2[i+1] - c1[i+1])**2)**0.5]
    if len(distances) > 0:
        return sum(distances) / len(distances)
    else:
        return float('nan')

def average_stable_distance(c1, c2):
    """
    Compute average euclidian distance (joint to joint) between two skeletons using only stable joints
    """
    distances = []
    for joint in STABLE_JOINTS:
        i = JOINTS[joint]
        if c1[i] != 0 and c1[i+1] != 0 and c2[i] != 0 and c2[i+1] != 0:
            distances += [((c2[i] - c1[i])**2 + (c2[i+1] - c1[i+1])**2)**0.5]
    if len(distances) > 0:
        return sum(distances) / len(distances)
    else:
        return float('nan')


def compute_angular_velocities(angles_t1, angles_t2):
    """
    Compute angular velocity for all angles (in rad/s)
    """
    dt = 1 / ut.FPS
    angular_velocities = {}
    for a in angles_t1:
        a1, a2 = angles_t1[a][0], angles_t2[a][0]
        angular_velocities[a] = (a2 - a1) / dt
    return angular_velocities


def update_histograms(frames, angles, angle_names, camera_id, gesture):
    ordered_frames = sorted(frames, key=lambda k: k['frame_index'])
    n_frames = len(frames)
    for frame in ordered_frames:
        frame_skeletons = []
        skeletons = frame['skeleton']
        for skeleton in skeletons:
            coordinates = skeleton['pose']
            if len(coordinates) != 0:
                frame_angles = compute_angles(angle_names, coordinates)
                for angle_name, (angle_value, _) in frame_angles.items():
                    if not math.isnan(angle_value):
                        angles[camera_id][gesture][angle_name] += [angle_value  * (180 / math.pi)]
    return angles


def interpolate(c1, c2, i1, i2):
    """
    Return interpolate of coordinates between given index
    """
    n = i2 - i1
    interpolated_coordinates = []
    for index in range(1, n):
        coordinates = []
        delta = index / n
        for i in range(0, len(c1), 2):
            x1 , y1 = c1[i], c1[i+1]
            x2 , y2 = c2[i], c2[i+1]
            if x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0:
                x_int = x1 * delta + (1 - delta) * x2
                y_int = y1 * delta + (1 - delta) * y2
            else:
                x_int, y_int = 0, 0
            coordinates += [x_int, y_int]
        interpolated_coordinates += [(index + i1, coordinates)]
    return interpolated_coordinates

