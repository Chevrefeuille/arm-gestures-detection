import csv
import json

group_annotations = {}

# create the JSON annotation file from the CSV file

with open('data/annotations_2.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        group_info = {}
        group_id = row[0]
        if group_id != '' and group_id != 'Pedestrian IDs': #first lines
            k = 0
            for camera_id in [1, 2, 4, 5]:
                camera_info = row[k+1:k+5]
                if camera_info != ['', '', '', '']:
                    group_info[camera_id] = camera_info
                k += 4
            intensity = row[17]
            group_annotations[group_id] = {
                'intensity': intensity,
                'camera_annotations': group_info
            }
        
with open('data/groups/interactions.json', 'w') as outfile:
    json.dump(group_annotations, outfile)
