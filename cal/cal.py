import json
import os
from PIL import Image

def cal():
    Note = open('preds.txt',mode='r')
    preds = Note.read()
    preds = json.loads(preds)
    Note.close()

    Note = open('truth.txt',mode='r')
    truth = Note.read()
    truth = json.loads(truth)
    Note.close()

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    exp = 1

    # num=0
    # for i in range(len(preds)):
    #     if truth[i] == 1:
    #         num+=1
    # print(num,len(preds))

    for i in range(len(preds)):
        if preds[i] == 1 and truth[i] == 1:
            tp += 1
        elif preds[i] == 1 and truth[i] == 0:
            fp += 1
        elif preds[i] == 0 and truth[i] == 1:
            fn += 1
        elif preds[i] == 0 and truth[i] == 0:
            tn += 1

    print(tp,fp,fn,tn)
    fnr = (fn + exp) / (tp + fn + exp)
    fpr = (fp + exp) / (fp + tn + exp)
    print("漏检率和误诊率分别为：",round(fnr,2), round(fpr,2))

def pred(path):
    preds = []       # 记录是否有预测框（漏检率误诊率用）

    for file_path in os.listdir(path):
        print(file_path)
        img = Image.open(file_path)
        clrs = img.getcolors()
        print(clrs) # 打印出来看 黄色(255, 255, 0)
        for clr in clrs:
            if clr[1]==(0, 0, 0): # 验证图像是否包含黄色
                preds.append(1)
            else:
                preds.append(0)

    # # 统计预测框（漏检率误诊率用）
    # Note=open('preds.txt',mode='w')
    # Note.write(str(preds)) #\n 换行符
    # Note.close()

if __name__ == "__main__":
    path = "/data/dataset/lhq/SINet/Result/polyp"
    pred(path)
    cal()