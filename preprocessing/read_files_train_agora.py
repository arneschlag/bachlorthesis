import pandas as pd
from tqdm import tqdm
import numpy as np
import json, glob
from PIL import Image
from collections import defaultdict
import time, cv2
from joblib import Parallel, delayed

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


train_dataset = ["train_0",
                 "train_1",
                 "train_2",
                 "train_3",
                 "train_4",
                 "train_5",
                 "train_6",
                 "train_7",
                 "train_8",
                 "train_9"]
annotation_id = 0
image_id = 0
annotations = []
loc_images = []
image_dir = {}

def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        print(f"Error: {e}")
        return None
def calculate_bbox(image_path):

    # Get pixel data
    img = cv2.imread(image_path)

    if img is not None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([1,1,1]), np.array([255,255,255]))

        seg_value = 255

        np_seg = np.array(mask)
        segmentation = np.where(np_seg == seg_value)

        # Bounding Box
        bbox = 0, 0, 0, 0
        if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))

            bbox = x_min, x_max, y_min, y_max
            test = cv2.rectangle(img, (x_min, y_min), (x_max ,y_max), (0, 255, 0), 2)

            return bbox

def find_semseg_img(fname, person_idx):
    folders = glob.glob('/mnt/data/AGORA/train_masks_3840x2160/train/*')
    file_id = int(fname.split('_')[-1])
    for folder in folders:
        if fname.startswith(folder.split('/')[-1]):
            for file in glob.glob(f'{folder}/*'):
                image_id, person_id = file.split('.')[0].split('_')[-2:]
                if int(image_id) == file_id and int(person_id) == person_idx+1:
                    return file
    # for img in semantics_train:
    #     if img.endswith(path):
    #         return img

def calculate(row, image_id, annotation_id):
    img_path = row["imgPath"]
    occ = row["occlusion"]
    gt_joints_2d_s = row['gt_joints_2d']
    gt_joints_3d_s = row['gt_joints_3d']
    annotations = []
    loc_images = []
    if img_path not in image_dir:
        image_dir[img_path] = image_id 
        size = get_image_size(f'/mnt/data/AGORA/imgs/{subset}/{img_path}')
        fname = img_path.split('.')[0]
        # if semseg_img:
        #    bboxs_real = calculate_bbox(semseg_img)
        if size:
            loc_images.append(dict(file_name=f'imgs/{subset}/{img_path}',
                            height=size[1],
                            width=size[0],
                            id=image_id)
            )
        else:
            return annotations, loc_images


    for person_idx in range(len(gt_joints_2d_s)):
        #semseg_img = '_'.join(fname.split('_')[:-1]) + "_mask_0" + fname.split('_')[-1] + f'_{person_idx+1:05d}.png'
        bboxs_real = None
        semseg_img = find_semseg_img(fname, person_idx)
        if semseg_img:
            bbox_real = calculate_bbox(semseg_img)
            keypoints_2d = gt_joints_2d_s[person_idx]
            keypoints_3d = gt_joints_3d_s[person_idx]
            occlusion = occ[person_idx]

            
            ys, xs = np.rot90(keypoints_2d)
            if bbox_real:
                bbox = dict(min_y=bbox_real[2],
                            min_x=bbox_real[0],
                            max_y=bbox_real[3],
                            max_x=bbox_real[1])
                # a = cv2.imread(f'/mnt/data/AGORA/imgs/{subset}/{img_path}', cv2.IMREAD_UNCHANGED)
                # a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            # else:
            #     bbox = dict(min_y=min(ys),
            #                 min_x=min(xs),
            #                 max_y=max(ys),
            #                 max_x=max(xs))
                
                width = bbox["max_x"]-bbox["min_x"]
                height = bbox["max_y"]-bbox["min_y"]
                #test = cv2.rectangle(a, (bbox_real[0], bbox_real[2]), (bbox_real[1], bbox_real[3]), (0, 255, 0), 2)

                anno = dict(area=float((bbox["max_x"]-bbox["min_x"])*(bbox["max_y"]-bbox["min_y"])),
                    category_id=0,
                    iscrowd=False,
                    image_id=image_dir[img_path],
                    bbox=[bbox["min_x"], bbox["min_y"], width, height],
                    id=annotation_id,
                    vis_ratio=1-(occlusion/100),
                    truncation=0,
                    keypoints=np.hstack([keypoints_2d,np.ones((45,1))*2]).flatten(),
                    keypoints_3d=keypoints_3d,
                    num_keypoints=45)
                annotation_id+=1
                annotations.append(anno)
        else:
            print('_'.join(fname.split('_')[:-1]) + "_mask_0" + fname.split('_')[-1] + f'_{person_idx+1:05d}.png')
    return annotations, loc_images

for subset in tqdm(train_dataset):
    df = pd.read_pickle(f'/mnt/data/AGORA/SMPL/{subset}_withjv.pkl')

    tasks = []
    for index, row in list(df.iterrows()):
        tasks.append((row, image_id, annotation_id))
        annotation_id += 1000
        image_id += 1
    results = Parallel(n_jobs=32)(
        delayed(calculate)(
            row, image_id, annotation_id
        ) for row, image_id, annotation_id in tasks
    )
    # for row, image_id, annotation_id in tasks:
    #     calculate(row, image_id, annotation_id)
    # print("done")
    for annos, locs in results:
        annotations += annos
        loc_images += locs
        
info = {'description': 'AGORA',
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
     "name": 'person'}
]
data = {"info": info, "licenses": licenses, "images": loc_images, "annotations": annotations, "categories": kia_categories}

print("Export")
with open(f'agora_train_all_n.json', 'w', encoding ='utf8') as json_file:
    json.dump(data, json_file, allow_nan=True, cls=NpEncoder)