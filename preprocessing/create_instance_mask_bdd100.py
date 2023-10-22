import json
import numpy as np
import pycocotools.mask as mask
import cv2

with open('/mnt/data/bdd100/bdd100k/labels/ins_seg/rles/ins_seg_val.json', 'r') as f:
    data = json.load(f)

out = "/mnt/data/bdd100/bdd100k/instance_imgs/val/"
for image in data['frames']:
    # create mask 
    mask_img = np.zeros((720,1280)).astype(np.uint8)
    i = 0
    for label in image['labels']:
        if label['category'] == 'pedestrian':
            mask_img += mask.decode(label['rle'])*(i+1)
            i+=1
    cv2.imwrite(f'{out}{image["name"].split(".")[0]}.png', mask_img)


print("a")