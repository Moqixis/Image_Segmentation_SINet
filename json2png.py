import json
import os
import os.path as osp

import warnings
import cv2

from labelme import utils

import numpy as np
import PIL.Image
import PIL.ImageDraw

def json2png(json_file, out_file):
    warnings.warn("This script is aimed to demonstrate how to convert the\n"
                  "JSON file to a single image dataset, and not to handle\n"
                  "multiple JSON files to generate a real-use dataset.")

    # freedom
    list_path = os.listdir(json_file)
    print('freedom =', json_file)
    for i in range(0, len(list_path)):
        print(i)
        path = os.path.join(json_file, list_path[i])
        if os.path.isfile(path):

            data = json.load(open(path))
            img = utils.img_b64_to_arr(data['imageData'])
            lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])

            mask = []
            mask.append((lbl).astype(np.uint8)) 
            mask = np.asarray(mask,np.uint8)
            mask = np.transpose(np.asarray(mask,np.uint8),[1,2,0])
            retval, im_at_fixed = cv2.threshold(mask[:,:,0], 0, 255, cv2.THRESH_BINARY)
            
            # out_dir = osp.basename(path).replace('.', '_')
            out_dir = osp.basename(path).split('.json')[0]
            save_file_name = out_dir
            # out_dir = osp.join(osp.dirname(path), out_dir)

            if not osp.exists(out_file + 'GT'):
                os.mkdir(out_file + 'GT')
            maskdir = out_file + 'GT'

            out_dir1 = out_file + 'Image'
            if not osp.exists(out_dir1):
                os.mkdir(out_dir1)

            PIL.Image.fromarray(img).save(out_dir1 + '/' + save_file_name + '.jpg')
            PIL.Image.fromarray(im_at_fixed).save(maskdir + '/' + save_file_name + '.png')

            print('Saved to: %s' % out_dir1)


# 图片重命名
def rename(path):
    # path = path + 'Image'
    path = path + 'GT'
    for file in os.listdir(path):
        name = file.split('.')[0]
        name = name.split('_')[1]
        name = name[5:]     # "image1"
        # ‘%04d’表示一共4位数，GT是png，Image是jpg
        os.rename(os.path.join(path, file), os.path.join(path, 'image' + '%04d' % int(name) + ".png"))   

if __name__ == '__main__':
    json_file = "/data/dataset/lhq/data/labels/" # 文件夹里面全是json
    out_file = "/data/dataset/lhq/SINet/DataSet/TrainDataset/"
    json2png(json_file, out_file)
    # rename(out_file)
