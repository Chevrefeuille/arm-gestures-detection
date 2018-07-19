import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import cv2
import h5py
import json
from tools import skeleton as sk
from scipy.signal import butter, lfilter, freqz

NUMBER_OF_FRAMES = 38370
FPS = 60

GROUND_TRUTH_FILE = 'data/ground_truth/trainvalRaw.mat'
GROUP_ANNOTATIONS_FILE = 'data/groups/groups_annotations.json'
GROUP_INTERACTIONS_FILE = 'data/groups/interactions.json'

THRESHOLD_DISTANCE = 2.5

D_BIN_SIZE = 0.1
D_MIN_TOLERABLE = 0
D_MAX_TOLERABLE = 2.5

D_MIN_PLOT = 0
D_MAX_PLOT = 2

VEL_BIN_SIZE = 3/60 
VEL_MIN_TOLERABLE = 0 # m/s
VEL_MAX_TOLERABLE = 3 # m/s

VEL_MIN_PLOT = 0
VEL_MAX_PLOT = 2.5

VELDIFF_BIN_SIZE = 0.5 / 60 
VELDIFF_MIN_TOLERABLE = 0 # m/s
VELDIFF_MAX_TOLERABLE = 0.5 # m/s

VELDIFF_MIN_PLOT = 0
VELDIFF_MAX_PLOT = 0.5

VELOCITY_THRESHOLD = 0.5 # m/s
DISTANCE_THRESHOLD = 2000 # mm


HISTOG_PARAM_TABLE = {
    'd': (D_MIN_TOLERABLE, D_MAX_TOLERABLE, D_BIN_SIZE),
    'v_G': (VEL_MIN_TOLERABLE, VEL_MAX_TOLERABLE, VEL_BIN_SIZE),
    'v_diff': (VELDIFF_MIN_TOLERABLE, VELDIFF_MAX_TOLERABLE, VELDIFF_BIN_SIZE),
}

PLOT_PARAM_TABLE = {
    'd': (D_MIN_PLOT, D_MAX_PLOT),
    'v_G': (VEL_MIN_PLOT, VEL_MAX_PLOT),
    'v_diff': (VELDIFF_MIN_PLOT, VELDIFF_MAX_PLOT),
}

PARAM_NAME_TABLE = {
    'v_G': r'$v_G$',
    'v_diff': r'$\omega$',
    'd': r'$\delta$'
}

PARAM_UNIT_TABLE = {
    'v_G': 'm/sec',
    'v_diff': 'm/sec',
    'd': 'm'
}

HOMOGRAPHY_MATRICES = {
    1: np.array([
        [-192.751647287874, -59.1901321344949, 0.0532577790056326],
        [452.643370680776, -22.3647433482908, 0.0296283312921158],
        [-26228.991498006, 5662.75738467188, 1]
    ]),
    2: np.array([
        [37.9862751408971, -18.4782128807507, 0.0331251620479919],
        [202.054236760698, -5.65823713960286, 0.00064290795166761],
        [-5261.18064851231, 1402.49892228599, 1]
    ]),
    4: np.array([
        [-218.745193143182, 7.12748948486925, 0.0058509289433717],
        [152.03818727057, -16.8708284611493, 0.114169654439258],
        [8669.38463329844, -1889.63691021174, 1]
    ]),
    5: np.array([
        [38.1258331882423, -22.4588156197574, 0.0269480937169095],
        [121.798278820536, -5.52107545545504, -0.00424362760836458],
        [580.967847048693, 515.343164446533, 1]
    ])
}

CAMERA_TVECS = {
    1: np.array([-55.3213480575146, 8.8662739619770, 7.08480108522491]),
    2: np.array([-30.0173026699373, 3.77174565384788, 31.4467932472112]),
    4: np.array([-32.5777798444836, 10.1383488912369, -9.55458173946362]),
    5: np.array([-1.72959400039486, -3.6170293785304, 33.2602545878358])
}

CAMERA_RVECS = {
    1: np.array([
        [-0.492642178017801, 0.870022905404954, -0.0190742892030141],
        [-0.205003721811655, -0.0947238758160186, 0.974166752354935],
        [0.845740597631003, 0.483825930910193, 0.225022910161696]
        ]),
    2: np.array([
        [0.0579414992425846, 0.997394058017463, 0.0429869247211089],
        [-0.200579993274932, -0.0305505412079555, 0.979200863321581],
        [0.97796239649761, -0.0653586831519556, 0.198287149289339]
    ]),
    4: np.array([
        [0.999600997054984, -0.0171760417660645, 0.0224238773616974],
        [-0.0162540781221277, 0.299484156755174, 0.953962811013638],
        [-0.023100901088092, -0.953946656497046, 0.299085480969949]
    ]),
    5: np.array([
        [0.132897206612524, 0.988585889956149, 0.0709666869326499],
        [-0.400280495614111, -0.0119682523592977, 0.916314512471775],
        [0.906704945009958, -0.150182219703508, 0.394121102682179]
    ])
}

def load_data():
    """ 
    Load pedestrians ground truth
    """
    with h5py.File(GROUND_TRUTH_FILE, 'r') as f:
        data = f['trainData'].value
    data = data.transpose()
    return data

def load_annotations():
    """ 
    Load pedestrians group annotations
    """
    with open(GROUP_ANNOTATIONS_FILE) as f:
        group_annotations = json.load(f)
    return group_annotations

def load_interactions():
    """ 
    Load pedestrians group interactions
    """
    with open(GROUP_INTERACTIONS_FILE) as f:
        group_interactions = json.load(f)
    return group_interactions

def compute_group_ids(group_annotations):
    """ 
    Return list of pedestrian ids that are in a group
    """
    group_ids = []
    for group_id, group_data in group_annotations.items():
        members = group_data['members']
        pedestrian_ids = list(map(int,(members)))
        for pid in pedestrian_ids:
            if pid not in group_ids:
                group_ids += [pid]
    return group_ids

def compute_angle(v1, v2):
    """
    Compute angle between vectors v1 and v2
    """
    if v1  != [0, 0] and v2 != [0, 0]:
        inner_product = np.vdot(v1, v2)
        cosang = np.dot(v1, v2) / (la.norm(v1) * la.norm(v2))
        sinang = la.norm(np.cross(v1, v2)) / (la.norm(v1) * la.norm(v2))
        angle = np.arctan2(sinang, cosang)
        # print(v1, v2, angle)
        return angle
    else:
        return float('nan')

WIDTH, HEIGHT = 1920, 1080 

def draw_info(frame, camera_id, frame_counter):
    """
    Display frame related information
    """
    # display camera id
    cv2.rectangle(frame, (10, 10), (250, 80), (255, 255, 255), -1)
    cv2.putText(frame, 'Camera {}'.format(camera_id), (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # display frame counter
    cv2.putText(frame, 'Frame {}'.format(frame_counter), (50, HEIGHT - 100),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return frame


def find_rois(frame_data, frame, draw=False):
    """
    Extract and display the bounding box for a frame
    """
    rois = []
    for data_line in frame_data:
        pedestrian_id = int(data_line[1])
        x1, y1 = map(int, [data_line[3], data_line[4]])
        x2, y2 = map(int, [x1 + data_line[5], y1 + data_line[6]])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)          
        x1, x2 = x1 / WIDTH, x2 / WIDTH
        y1, y2 = y1 / HEIGHT, y2 / HEIGHT
        rois.append([(x1, y1), (x2, y2)])
    return rois, frame

def find_feet_coordinates(frame_data, frame, draw=False):
    """
    Extract and display the feet coordinates
    """
    coordinates = []
    for data_line in frame_data:
        pedestrian_id = int(data_line[1])
        x, y = data_line[9], data_line[10]
        cv2.circle(frame, (int(x * WIDTH), int(y * HEIGHT)), 3, (0, 255, 0), -1)      
        coordinates += [(x, y)]
    return coordinates, frame


def update_world_feet_coordinates(frame_data, feet_coordinates):
    """
    Extract the world feet coordinates
    """
    for data_line in frame_data:
        pedestrian_id = int(data_line[1])
        x, y = data_line[7], data_line[8]
        feet_coordinates[pedestrian_id] += [(x, y)]
    return feet_coordinates


def find_appearance(data):
    """
    Find the ranges of appearance for the data of a given pedestrian
    """
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
    """
    Find the range of appearance of a group of pedestrian
    """
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


def find_appearances(data):
    """
    Find the various range of appearances of a pedestrian given its ground truth data
    """
    appearances = {}
    for camera_id in [1, 2, 4, 5]:
        camera_data = data[data[:, 0] == int(camera_id)]
        if camera_data.size != 0:
            frames = camera_data[:,2]
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
                appearances[camera_id] = frame_ranges
    return appearances


def median_filter(signal, size):
    """
    Filter the signal with a median filter of given size
    """
    n = len(signal)
    augmented_signal = [signal[0]] + signal + [signal[-1]] # to prevent boundary issues
    filtered_signal = []
    for i in range(0, n):
        portion = augmented_signal[i:i+size]
        filtered_signal += [sorted(portion)[size//2]]
    return filtered_signal


def update_height_deltas(height_deltas, feet_coord, coordinates):
    """
    Compute height from joints to feet
    """
    for j in height_deltas:
        j_index =  sk.JOINTS[j]
        if coordinates[2 * j_index + 1] != 0:
            delta_h = feet_coord[1] - coordinates[2 * j_index + 1] 
        else:
            delta_h = float('nan')
        height_deltas[j] += [delta_h]
    return height_deltas


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def is_inside_roi(p, roi):
    """
    Check if a point is inside a given rectangle
    """
    x1, y1 = roi[0]
    x2, y2 = roi[1]
    return p[0] >= x1 and p[0] <= x2 and p[1] >= y1 and p[1] <= y2

def euclidian_distance(p, q):
    """
    Compute euclidian distance between 2 points
    """
    return ((p[0] - q[0])**2 + (p[1] - q[1]))**0.5


def compute_observables(world_feet_coords):
    """
    Compute a set of observables for a group of pedestrians
    """
    vx, vy = compute_velocities(world_feet_coords)
    if len(world_feet_coords) == 2:
        pids = list(world_feet_coords.keys())
        # interpersonal distance
        d = []
        traj_A = world_feet_coords[pids[0]]
        traj_B = world_feet_coords[pids[1]]
        for i in range(min(len(traj_A), len(traj_B))):
            distance = euclidian_distance(traj_A[i], traj_B[i])
            if distance >= 0.5 and distance <= THRESHOLD_DISTANCE:
                d += [distance]
            # print(euclidian_distance(traj_A[i], traj_B[i]))
        d = np.array(d)
        # group_velocity
        vx_A, vy_A = np.array(vx[pids[0]]), np.array(vy[pids[0]])
        vx_B, vy_B = np.array(vx[pids[1]]), np.array(vy[pids[1]])
        vx_G, vy_G = (vx_A + vx_B) / 2, (vy_A + vy_B) / 2
        v_G = (vx_G**2 + vy_G**2)**.5
        v_G = v_G[v_G > 0.5]
        # velocity difference
        vx_diff = (vx_A - vx_B) / 2
        vy_diff = (vy_A - vy_B) / 2
        v_diff = (vx_diff**2 + vy_diff**2)**.5
        v_diff = v_diff[v_diff < 0.5]
        return {
            'd': d,
            'v_G': v_G,
            'v_diff': v_diff
        }
    else:
        return []

def update_observables(world_feet_coords, histograms, intensity):
    """
    Update the observables
    """
    new_observables = compute_observables(world_feet_coords)
    for o in new_observables:
        histograms[intensity][o] = np.concatenate((histograms[intensity][o], new_observables[o]))
    return histograms

def compute_mean_observables(observables):
    """
    Compute the mean value for the observables
    """
    mean_values = {}
    for o, values in observables.items():
        mean_values[o] = np.mean(values)
    return mean_values

def update_mean_observables(mean_observables, observables, intensity):
    """
    Update the mean values for the observables
    """
    new_mean_observables = compute_mean_observables(observables)
    for o in new_mean_observables:
        mean_observables[intensity][o] = np.append(mean_observables[intensity][o], new_mean_observables[o])
    return mean_observables

def compute_velocities(world_feet_coords):
    """
    Compute the velocity of a pedestrian
    """
    dt = 1 / FPS
    vx, vy = {}, {}
    for pid in world_feet_coords:
        vx[pid] = []
        vy[pid] = []
    for pid, traj in world_feet_coords.items():
        for i in range(len(traj) - 1):
            vx[pid] += [(traj[i+1][0] - traj[i][0]) / dt]
            vy[pid] += [(traj[i+1][1] - traj[i][1]) / dt]
    return vx, vy


def compute_histogram(obs, obs_data):
    """
    Compute the histogram of the given observable data
    """
    (min_bound, max_bound, bin_size) = HISTOG_PARAM_TABLE[obs]
    n_bins = round((max_bound - min_bound) / bin_size) + 1
    edges = np.linspace(min_bound, max_bound, n_bins)
    histog = np.histogram(obs_data, edges)
    return histog[0]

def compute_pdf(obs, histogram):
    """
    Compute the PDF of the given observable histogram
    """
    (_, _, bin_size) = HISTOG_PARAM_TABLE[obs]
    pdf = histogram / sum(histogram) / bin_size
    return pdf

def get_edges(obs):
    """
    Compute the abscissa value to plot the PDF of the given observable parameter
    """
    (min_bound, max_bound, bin_size) = HISTOG_PARAM_TABLE[obs]
    return np.arange(min_bound, max_bound, bin_size)

def plot_pdf(pdfs, intensities):
    """
    Plot the histograms for the given observables
    """
    plt.rcParams['grid.linestyle'] = '--'
    for o in pdfs:
        edges = get_edges(o)
        for i in intensities:
            plt.plot(edges, pdfs[o][i], label=i, linewidth=3)
        plt.xlabel('{}({})'.format(PARAM_NAME_TABLE[o], PARAM_UNIT_TABLE[o]))
        plt.ylabel('p({})'.format(PARAM_NAME_TABLE[o]))
        plt.legend()
        plt.xlim(PLOT_PARAM_TABLE[o])
        plt.grid()
        plt.show()

def save_pdf(pdfs, intensities):
    """
    Plot the histograms for the given observables
    """
    for o in pdfs:
        edges = get_edges(o)
        for i in intensities:
            file_name = 'data/pdfs/{}_{}.txt'.format(o, i)
            np.savetxt(file_name, np.c_[edges, pdfs[o][i]])
      