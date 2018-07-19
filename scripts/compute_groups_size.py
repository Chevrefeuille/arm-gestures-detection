import scipy.io

camera_ids = [1, 2, 4, 5]

for camera_id in camera_ids:

    group_annotation_file = 'groups/camera{}/groups_predicted.mat'.format(str(camera_id))

    f = scipy.io.loadmat(group_annotation_file)
    groups = f['groups']

    groups_size_dict = {}

    seen_groups = []

    for line in groups:
        ids = list(line[3][0])
        if ids not in seen_groups:
            group_size = len(ids)
            if group_size not in groups_size_dict:
                groups_size_dict[group_size] = 1
            else:
                groups_size_dict[group_size] += 1
            seen_groups += [ids]
    
    group_str = ', '.join(['{} groups of size {}'.format(n, size) for size, n in groups_size_dict.items()])
    print('For camera {}: {}'.format(camera_id, group_str))