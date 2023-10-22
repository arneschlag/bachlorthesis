"""some persons contained no 3d keypoint information, remove those from the dataset ...
"""

import json

# '/home/scl9hi/mount_mfsd/02_stud/01_schlag/01_datasets/01_ki_a/3d/all_test_3d.json'
# '/home/scl9hi/mount_mfsd/02_stud/01_schlag/01_datasets/01_ki_a/3d/all_train_3d.json'
# '/home/scl9hi/mount_mfsd/02_stud/01_schlag/01_datasets/01_ki_a/3d/all_validaa_3d.json'

with open('/data/projects/MFSD/02_stud/01_schlag/01_datasets/01_ki_a/3d/all_train_3d.json', 'r') as data_file:
    json_data = data_file.read()

data = json.loads(json_data)

# remove image_id
image_ids = []
i=0
for anno in data['annotations']:
    # remove every pedestrian, that has no 3d keypoints
    if len(anno["keypoints_3d"]) == 0 and anno["category_id"] == 2:
        i+=1
        image_ids.append(anno["image_id"])
# delete images from training
for index in sorted(image_ids, reverse=True):
    del data["images"][index]

# delete all annotations related to those images ...
i = 0
data_ids = []
for anno in data["annotations"]:
    if anno["image_id"] in image_ids:
        data_ids.append(i)
        i+=1

for index in sorted(data_ids, reverse=True):
    del data["annotations"][index]

with open('/data/projects/MFSD/02_stud/01_schlag/01_datasets/01_ki_a/3d/all_train_3d_fixed.json', 'w', encoding ='utf8') as json_file:
        json.dump(data, json_file, allow_nan=True)
