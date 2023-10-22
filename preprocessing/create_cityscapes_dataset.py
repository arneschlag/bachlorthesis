import scipy.io, json
mat = scipy.io.loadmat('/mnt/data/CityPersons/annotations/anno_train.mat')

import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

train_or_val = "train"
coco_dict = {
    'info': {
        'description': f'CityPersons Dataset',
        'url': 'https://github.com/cvgroup-njust/CityPersons',
        'version': '1.0',
        'year': 2020,
        'contributor': 'Arne Schlag',
        'date_created': '2023/07/18',
    },
    'licences': [{
        'url': 'http://creativecommons.org/licenses/by-nc/2.0',
        'id': 2,
        'name': 'Attribution-NonCommercial License'
    }],
    'images': [],
    'annotations': [],
    'categories': [{
        'supercategory': 'person',
        'id': 1,
        'name': 'person',
        'keypoints': None,
        'skeleton': None
    }]
}
i = 0
for row in mat['anno_train_aligned'][0]:
    image = row[0]
    for annotation in image:
        city, im_name, bbs = annotation  #in each cell, there are three fields: city_name; im_name; bbs (bounding box annotations)
        city, im_name, bbs = city[0], str(im_name[0]), bbs
        img_name_wh = im_name.split('.')[0]
        camera = img_name_wh.split('_')[-1]
        full_image_path = camera+"/"+train_or_val+"/"+city+"/"+im_name.split('.')[0]+".np.lz4"
        coco_dict['images'].append({
            'license': 5,
            'file_name': full_image_path,
            'height': 1080,
            'width': 1920,
            'date_captured': '2018-01-28 00:00:00',
            'id': i
        })

        # now process the annoations
        for bbox in bbs:
            class_label, x1,y1,w,h, instance_id, x1_vis, y1_vis, w_vis, h_vis = bbox
            if class_label == 1:
                if 0 in [w, y1, x1, h]:
                    vis_ratio = 0
                else:
                    vis_ratio = float((w_vis/w)*(y1_vis/y1)*(x1_vis/x1)*(h_vis/h))
                annotation = dict(area=int(w)*int(h),
                    category_id=2, # 2 is person
                    iscrowd=False,
                    image_id=i,
                    id=instance_id,
                    bbox=[x1,y1, w, h],
                    vis_ratio=vis_ratio
                )
                coco_dict['annotations'].append(annotation)
        i+=1
print(i)
#with open('train_citypersons.json', "w") as f:
#    json.dump(coco_dict, f, cls=NpEncoder)