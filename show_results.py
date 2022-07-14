from glob import glob
from PIL import Image
import numpy as np
import os
import os.path as osp
from matplotlib import pyplot as plt

def show_result_random():
    image_path = '/data/dataset/lhq/SINet/DataSet/TestDataset/polyp/Image/'
    gt_path = '/data/dataset/lhq/SINet/DataSet/TestDataset/polyp/GT/'
    result_path = '/data/dataset/lhq/SINet/Result/polyp/'
    out_path = '/data/dataset/lhq/SINet/Result/show_result/'
    if not osp.exists(out_path):
        os.makedirs(out_path)
    test_num = len(glob(image_path+'/*'))
    for epoch in range(5):
        rand_index = []
        while len(rand_index) < 3:
            np.random.seed()
            temp = np.random.randint(0, test_num, 1)
            rand_index.append(temp)
        rand_index = np.array(rand_index).squeeze()
        fig, ax = plt.subplots(3, 3, figsize=(18, 18))
        for i, index in enumerate(rand_index):
            # # 计算dice系数
            # fz = 2 * np.sum(mask.squeeze() * x_label[index].squeeze())
            # fm = np.sum(mask.squeeze()) + np.sum(x_label[index].squeeze())
            # dice = fz / fm
            image_index = [name for name in os.listdir(image_path) if name[-8:]=='%04d'%index+'.jpg']
            img = image_path + image_index[0]
            image = np.array(Image.open(img), dtype='float32') / 255
            gt_index = [name for name in os.listdir(gt_path) if name[-8:]=='%04d'%index+'.png']
            img = gt_path + gt_index[0]
            gt = np.array(Image.open(img), dtype='float32') / 255
            result_index = [name for name in os.listdir(result_path) if name[-8:]=='%04d'%index+'.png']
            img = result_path + result_index[0]
            sinet_result = np.array(Image.open(img), dtype='float32') / 255
            ax[i][0].imshow(image.squeeze())
            ax[i][0].set_title('image', fontsize=20)
            ax[i][1].imshow(gt.squeeze())
            ax[i][1].set_title('gt', fontsize=20)
            ax[i][2].imshow(sinet_result.squeeze())
            ax[i][2].set_title('SINet output', fontsize=20)  # 设置title SINet
        fig.savefig(out_path + '/show%d_%d_%d.png' % (rand_index[0], rand_index[1], rand_index[2]),
                    bbox_inches='tight', pad_inches=0.1)  # 保存绘制的图片
        print('finished epoch: %d' % epoch)
        plt.close()

def show_result(image_path, result_path, out_path):
    if not osp.exists(out_path):
        os.makedirs(out_path)
    test_num = len(glob(image_path+'/*'))
    for epoch in range(test_num):
        index = epoch + 1 
        fig, ax = plt.subplots(2, 1, figsize=(18, 18))

        image_index = [name for name in os.listdir(image_path) if name[-8:]=='%04d'%index+'.jpg']
        img = image_path + image_index[0]
        image = np.array(Image.open(img), dtype='float32') / 255
        result_index = [name for name in os.listdir(result_path) if name[-8:]=='%04d'%index+'.png']
        img = result_path + result_index[0]
        sinet_result = np.array(Image.open(img), dtype='float32') / 255
        ax[0].imshow(image.squeeze())
        ax[0].set_title('image', fontsize=20)
        ax[1].imshow(sinet_result.squeeze())
        ax[1].set_title('SINet output', fontsize=20)  # 设置title SINet

        fig.savefig(out_path + '/show%04d.png' % (index),
                    bbox_inches='tight', pad_inches=0.1)  # 保存绘制的图片
        print('finished epoch: %d' % epoch)
        plt.close()

if __name__ == '__main__':
    # show_result_random()  # 随机显示
    image_path = '/data/dataset/lhq/SINet/DataSet/TestDataset/04130102/Image/'
    result_path = '/data/dataset/lhq/SINet/Result/04130102/'
    out_path = '/data/dataset/lhq/SINet/Result/show_result/04130102/'
    show_result(image_path, result_path, out_path)  # 全部显示
