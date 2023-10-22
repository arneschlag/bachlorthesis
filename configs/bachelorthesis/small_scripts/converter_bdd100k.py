import json
import numpy as np

with open('/mnt/data/bdd100/bdd100k/labels/bdd100k_labels_images_val.json', 'r') as data_file:
    json_data = data_file.read()

with open('/mnt/data/bdd100/bdd100k/labels/pose_21/pose_val.json', 'r') as data_file:
    pose_data = data_file.read()

pose_data = json.loads(pose_data)
bd100k_names = {
    0: "head",
    1: "neck",
    2: "right_shoulder",
    3: "right_elbow",
    4: "right_wrist",
    5: "left_shoulder",
    6: "left_elbow",
    7: "left_wrist",
    8: "right_hip",
    9: "right_knee",
    10: "right_ankle",
    11: "left_hip",
    12: "left_knee",
    13: "left_ankle",
    14: "right_hand",
    15: "left_hand",
    16: "right_foot",
    17: "left_foot",
}

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def gen_anno(label):
    if label['category'] in ['pedestrian', 'person', 'rider']:
        width=abs(label['box2d']['x1']-label['box2d']['x2'])
        height=abs(label['box2d']['y1']-label['box2d']['y2'])
        center_x = float(label['box2d']['x1'])
        center_y = float(label['box2d']['y1'])

        # v = visible = 2
        # c = covered = 1
        # n = not annotated ... =0
        vis_conv = dict(V=2,
                        C=1,
                        N=0)
        vis_ratio = 1
        truncation = 0
        if label['attributes']:
            if label['attributes']['occluded']:
                vis_ratio = 0 if label['attributes']['occluded'] else 1
            if label['attributes']['truncated']:
                truncation = 1 if label['attributes']['truncated'] else 0

       
        kp_2d = []
        if 'graph' in label:
            if label['graph']['nodes']:
                converter = dict((label["category"],i) for i, label in enumerate(label['graph']['nodes']))
                for kp in bd100k_names.values():
                    entry = label['graph']['nodes'][converter[kp]]
                    vis = vis_conv[entry['visibility']]

                    if vis < 2:
                        vis_ratio-=(1/18)

                    x, y = entry['location']
                    if x > 1280 or x < 0 or y > 720 or y < 0:
                        truncation+=(1/18)
                    kp_2d += [x,y,vis]

        cat_ids = {
            "pedestrian": 0,
            "rider": 1,
            "person": 2,
            "bike": 3, 
            "motor":4
        }

        anno = dict(area=float(width*height),
            category_id=cat_ids[label['category']],
            iscrowd=False,
            image_id=f_id,
            bbox=[center_x, center_y, width, height],
            id=anno_id,
            vis_ratio=vis_ratio,
            truncation=truncation,
            keypoints=kp_2d,
            num_keypoints=18)
        annoatations.append(anno)
        return anno

data = json.loads(json_data)
loc_images = []
annoatations = []
anno_id = 0

lookup_extra_data = dict((item["name"], i) for i, item in enumerate(data))
categories = []
for f_id, frame in enumerate(pose_data['frames']):
    extra = data[lookup_extra_data[frame['name']]]
    loc_images.append(dict(file_name="val/"+frame['name'],
                           height=720,
                           width=1280,
                           id=f_id,
                           attributes=extra['attributes']))
    if frame['labels']:
        for label in frame['labels']:
            if label['category'] not in categories:
                categories.append(label['category'])
            anno = gen_anno(label)
            if anno:
                annoatations.append(anno)
                anno_id+=1
    if extra['labels']:
        for label in extra['labels']:
            if label['category'] not in categories:
                categories.append(label['category'])
            anno = gen_anno(label)
            if anno:
                annoatations.append(anno)
                anno_id+=1
info = {'description': 'BDD100k',
    'url': 'http://myurl.org',
    'version': '1.0',
    'year': 2023,
    'contributor': 'Robert Bosch GmbH',
    'date_created': '2023/05/13'
}
licenses = [
    {'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 
    'id': 1, 
    'name': 'Edit this license'}
]
kia_categories = [
    {'supercategory': 'object',
     "id": 0, 
     "name": 'pedestrian'},
     {'supercategory': 'object',
     "id": 1, 
     "name": 'bdd_rider'},
     {'supercategory': 'object',
     "id": 2, 
     "name": 'bdd_person'},
     {'supercategory': 'object',
     "id": 3, 
     "name": 'bdd_bike'},
     {'supercategory': 'object',
     "id": 4, 
     "name": 'bdd_motor'}
]
print(categories)
data = {"info": info, "licenses": licenses, "images": loc_images, "annotations": annoatations, "categories": kia_categories}
import json
with open('bdd100k_val_with_all.json', 'w', encoding ='utf8') as json_file:
    json.dump(data, json_file, allow_nan=True, cls=NpEncoder)
