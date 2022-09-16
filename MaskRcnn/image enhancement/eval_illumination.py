
'''
----张晓明----2022.6
----图像增强---

'''

import os
import cv2
import numpy as np

import skimage
from skimage import io
from skimage import transform as tf
from skimage.filters.rank import mean_bilateral
from skimage import morphology

from PIL import Image
from PIL import ImageEnhance

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict

import argparse

from model_illNet import illNet

# For parsing commandline arguments  光照增强

# parser = argparse.ArgumentParser()
# #输入的图片
# parser.add_argument("--imgPath", type=str, default=r'E:\output_title\crop0.png')
# #输出的图片
# parser.add_argument("--savPath", type=str, default=r'E:\output_title\crop00.png')
# #模型
# parser.add_argument("--modelPath", type=str, default='./model_illNet.pkl')
# args = parser.parse_args()

def preProcess(img):
    
    img[:,:,0] = mean_bilateral(img[:,:,0], morphology.disk(20), s0=10, s1=10)
    img[:,:,1] = mean_bilateral(img[:,:,1], morphology.disk(20), s0=10, s1=10)
    img[:,:,2] = mean_bilateral(img[:,:,2], morphology.disk(20), s0=10, s1=10)
    
    return img
    
#裁剪图片
def padCropImg(img):
    img = img[:, :, :3]

    H = img.shape[0]
    W = img.shape[1]
    patchRes = 128
    pH = patchRes
    pW = patchRes
    # pZ = patchRes
    ovlp = int(patchRes * 0.125)

    padH = (int((H - patchRes) / (patchRes - ovlp) + 1) * (patchRes - ovlp) + patchRes) - H
    padW = (int((W - patchRes) / (patchRes - ovlp) + 1) * (patchRes - ovlp) + patchRes) - W
    # padZ = (int((Z - patchRes)/(patchRes - ovlp) + 1) * (patchRes - ovlp) + patchRes) - Z

    padImg = cv2.copyMakeBorder(img, 0, padH, 0, padW, cv2.BORDER_REPLICATE)
    # print(padImg.shape[2])
    ynum = int((padImg.shape[0] - pH) / (pH - ovlp)) + 1
    xnum = int((padImg.shape[1] - pW) / (pW - ovlp)) + 1


    totalPatch = np.zeros((ynum, xnum, patchRes, patchRes, 3), dtype=np.uint8)

    # for k in range(0, znum):
    for j in range(0, ynum):
        for i in range(0, xnum):
            x = int(i * (pW - ovlp))
            y = int(j * (pH - ovlp))
            # z = int(k * (pZ - ovlp))
            totalPatch[j, i] = padImg[y:int(y + patchRes), x:int(x + patchRes)]

    return totalPatch



def illCorrection(modelPath, totalPatch):
    
    model = illNet()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if torch.cuda.is_available():
        model = model.cuda()
        
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(modelPath))
    else:

        state_dict = torch.load(modelPath, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        #model.load_state_dict(new_state_dict)
        model.load_state_dict(new_state_dict,False)

    model.eval()

    totalPatch = totalPatch.astype(np.float32)/255.0

    ynum = totalPatch.shape[0]
    xnum = totalPatch.shape[1]
    #scal = totalPatch.shape[2]

    totalResults = np.zeros((ynum, xnum, 128,128, 3), dtype = np.float32)

    for j in range(0, ynum):
        for i in range(0, xnum):
            #for k in range(0,scal):

            patchImg = totalPatch[j, i]
            patchImg = transform(patchImg)

            if torch.cuda.is_available():
                patchImg = patchImg.cuda()

            patchImg = patchImg.view(1,3,128,128)
            patchImg = Variable(patchImg)

            output = model(patchImg)
            output = output.permute(0, 2, 3, 1).data.cpu().numpy()[0]

            output[output>1] = 1
            output[output<0] = 0
            output = output*255.0
            output = output.astype(np.uint8)

            totalResults[j,i] = output

    return totalResults

def composePatch(totalResults):

    ynum = totalResults.shape[0]
    xnum = totalResults.shape[1]
    patchRes = totalResults.shape[2]
    
    ovlp = int(patchRes * 0.125)
    step = patchRes - ovlp
    
    resImg = np.zeros((patchRes + (ynum - 1) * step, patchRes + (xnum - 1) * step, 3), np.uint8)
    
    for j in range(0, ynum):
        for i in range(0, xnum):
            
            sy = int(j*step)
            sx = int(i*step)

            resImg[sy:(sy + patchRes), sx:(sx + patchRes)] = totalResults[j, i]
    return resImg

def postProcess(img):
    
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(img)
    factor = 2.0
    img = enhancer.enhance(factor)

    return img
if __name__ == '__main__':
    modelPath = r'E:\python\MaskRcnn\image enhancement\model_illNet.pkl'
    project_dir = os.path.dirname(os.path.abspath(__file__))
    input = os.path.join(project_dir, r'E:\python\img\img')
    # 切换目录
    os.chdir(input)
    # 遍历目录下所有的文件
    #照片索引
    i = 1
    for image_name in os.listdir(os.getcwd()):
        #img = Image.open(os.path.join(input, image_name))
        img_path = os.path.join(input, image_name)
        print(img_path)
        img = io.imread(img_path)
        img = preProcess(img)
        totalPatch = padCropImg(img)
        totalResults = illCorrection(modelPath, totalPatch)
        resImg = composePatch(totalResults)
        resImg = postProcess(resImg)
        resImg.save(r'E:\output_title\{}.png'.format(i))
        i += 1

    # img = io.imread(args.imgPath)
    # img = preProcess(img)
    # totalPatch = padCropImg(img)
    # totalResults = illCorrection(args.modelPath, totalPatch)
    # resImg = composePatch(totalResults)
    # resImg = postProcess(resImg)
    # resImg.save(args.savPath)
