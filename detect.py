import time
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc  # NOTES: pip install scipy == 1.2.2 (prerequisite!)
from Src.SINet import SINet_ResNet50
from Src.utils.Dataloader import test_dataset
from Src.utils.trainer import eval_mae, numpy2tensor


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                    default='/data/models/lhq/SINet/polyp/SINet_40.pth')
parser.add_argument('--test_save', type=str,
                    default='/data/dataset/lhq/SINet/Result/')
opt = parser.parse_args()

model = SINet_ResNet50().cuda()
model.load_state_dict(torch.load(opt.model_path))
model.eval()

for dataset in ['04130102']:
    save_path = opt.test_save + dataset + '/'
    os.makedirs(save_path, exist_ok=True)
    # NOTES:
    #  if you plan to inference on your customized dataset without grouth-truth,
    #  you just modify the params (i.e., `image_root=your_test_img_path` and `gt_root=your_test_img_path`)
    #  with the same filepath. We recover the original size according to the shape of grouth-truth, and thus,
    #  the grouth-truth map is unnecessary actually.
    test_loader = test_dataset(image_root='/data/dataset/lhq/SINet/DataSet/TestDataset/{}/Image/'.format(dataset),
                               gt_root='/data/dataset/lhq/SINet/DataSet/TestDataset/{}/GT/'.format(dataset),
                               testsize=opt.testsize)
    img_count = 1
    fps_list = [] # 计算平均fps
    for iteration in range(test_loader.size):
        t1 = time.time()
        # load detect data
        image, name = test_loader.load_detect_data()
        image = image.cuda()
        # inference
        _, cam = model(image)
        # reshape and squeeze 修改size为原图大小
        cam = F.upsample(cam, size=(1080, 1160), mode='bilinear', align_corners=True)
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        misc.imsave(save_path+name, cam)

        # coarse score
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{})'.format(dataset, name, img_count,
                                                                           test_loader.size))
        fps_list.append(1 / (time.time() - t1)) # 计算平均fps
        img_count += 1

print('mean FPS:', sum(fps_list)/len(fps_list))
print("\n[Congratulations! Testing Done]")
