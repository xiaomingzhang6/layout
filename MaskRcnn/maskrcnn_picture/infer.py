#张晓明
#2022.7

'''
infer.py  批量处理照片
输入：照片文件夹
输出：图片切割后的照片 output1  output2 .....
     切割的照片  crop0  crop1  ...
'''

import os
import sys
import requests
import random
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import transforms
import argparse
from PIL import Image
import cv2
import numpy as np
from shutil import copy
from utils import (
    overlay_ann,
    # overlay_mask,
    show
)

seed = 1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#python infer.py --image_path img/4.png --output_path output/

CATEGORIES2LABELS = {
    0: "bg",
    1: "picture",
}
SAVE_PATH = "output/"

#MODEL_PATH = "./model_196000.pth"   #模型参数
#MODEL_PATH = "./T/my_model_picture2.pth"

MODEL_PATH =r'E:\python\MaskRcnn\maskrcnn_picture\T\my_model_picture2.pth'

#重写模型
def get_instance_segmentation_model(num_classes):
    #mask和数量一致
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)   #获取模型架构
    #print("=========model",model)

    # (cls_score): Linear(in_features=1024, out_features=91, biasbias=True)
    '''  in_features：每个输入（x）样本的特征的大小
    　　out_features：每个输出（y）样本的特征的大小'''

    in_features = model.roi_heads.box_predictor.cls_score.in_features   #获取模型的输入特征数in_features=1024
    print("==========in", model.roi_heads.box_predictor)
    '''==========in FastRCNNPredictor(
  (cls_score): Linear(in_features=1024, out_features=91, bias=True)
  (bbox_pred): Linear(in_features=1024, out_features=364, bias=True)
   )'''
   #修改模型的框架参数
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print("model.roi_heads.box_predictor===",model.roi_heads.box_predictor)
    '''经过FastRCNNPredictor后输出的特征数由91变为了6
    model.roi_heads.box_predictor=== FastRCNNPredictor(
      (cls_score): Linear(in_features=1024, out_features=6, bias=True)
      (bbox_pred): Linear(in_features=1024, out_features=24, bias=True)
      )
      '''

    print("model.roi_heads.mask_predictor==",model.roi_heads.mask_predictor)

    ''' 一开始的模型mask的参数  输出的类别为91个
    model.roi_heads.mask_predictor== MaskRCNNPredictor(
  (conv5_mask): ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))
  (relu): ReLU(inplace=True)
  (mask_fcn_logits): Conv2d(256, 91, kernel_size=(1, 1), stride=(1, 1))
  )
  '''
    #获取输入特征的，mask
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    print(in_features_mask)
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(    #把模型的参数改为

        in_features_mask,
        hidden_layer,
        num_classes
    )
    '''经过MaskRCNNPredictor 改成后的 输出的类别为六个
    MaskRCNNPredictor(
(conv5_mask): ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))
(relu): ReLU(inplace=True)
(mask_fcn_logits): Conv2d(256, 6, kernel_size=(1, 1), stride=(1, 1))
)'''
    print(model.roi_heads.mask_predictor)
    '''
    =============== MaskRCNN(
   (transform): GeneralizedRCNNTransform(
      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      Resize(min_size=(800,), max_size=1333, mode='bilinear')
    )
 
    '''
    return model
#适当增加切割图片大小
def site_img(box,images):
    if box[0] - 10 < 0:
        x_1 = box[0]
    else:
        x_1 = box[0] - 10
    if box[1] - 10 < 0:
        y_1 = box[1]
    else:
        y_1 = box[1] - 10
    if box[2] + 10 > images.width:
        x_2 = box[2]
    else:
        x_2 = box[2] + 10
    if box[3] + 10 > images.height:
        y_2 = box[3]
    else:
        y_2 = box[3] + 10

    box = (x_1, y_1, x_2, y_2)

    return box
#切割照片
def crop_img(box, images, idx,i):
    #保存照片路径
    path = r'E:\python\MaskRcnn\maskrcnn_picture\output/{0}_{1}.png'.format(i, idx)
    # 获取坐标
    x_1 = box[0]
    y_1 = box[1]
    x_2 = box[2]
    y_2 = box[3]
    box = (x_1, y_1, x_2, y_2)
    im2 = images.crop(box)
    im2.save(path)

'''把输出的图片复制到指定文件夹里'''
def move_img(i, count):
    #保存切割照片的目的路径
    dst_path = r'C:\Users\张晓明\Desktop\crop_img\{}'.format(i)
    #保存输出照片的目的路径
    output_path =r'E:\python\img\output_img' + '\\' + (str(i).zfill(7)) + '.png'
    #保存根路径
    dst_output_path = r'E:\python\MaskRcnn\maskrcnn_picture\output\delete_img.png'
    #创建文件夹
    os.mkdir(dst_path)
    '''把布局分析delete_img复制到文件夹'''
    copy(dst_output_path, output_path)
    for j in range(count):
        src_path = r'E:\python\MaskRcnn\maskrcnn_picture\output\{0}_{1}.png'.format(i, j)
        copy(src_path, dst_path)

def picture_main(img_path, i):
    num_classes = 2
    model = get_instance_segmentation_model(num_classes)

    #model.cuda()
    model.to(device)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default = 'None',
        type = str,
        help = "model checkpoint directory"
    )

    args = parser.parse_args()

    if os.path.exists(MODEL_PATH):

        checkpoint_path = MODEL_PATH
    else:
        checkpoint_path = args.model_path

    print(checkpoint_path)

    assert os.path.exists(checkpoint_path)    #找到模型参数路径  往下执行
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    #model.load_state_dict(torch.load(checkpoint)) #表示将训练好的模型参数重新加载至网络模型中

    model.eval()
    image_path = img_path
    print("================", image_path)

    image = cv2.imread(image_path)
    # rat = 1000 / image.shape[0]
    #
    # image = cv2.resize(image, None, fx=rat, fy=rat) #如果fx=1.44，则将原图片的x轴缩小为原来的1.44倍，将y轴缩小为原来的1.44倍
    '''transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起
    transforms. ToTensor()将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，
    其将每一个数值归一化到[0,1]，其归一化方法比较简单，直接除以255即可
    '''
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    image = transform(image)
    #print("========image.to(device)",image.to(device))
    with torch.no_grad():
        prediction = model([image.to(device)])
        print(prediction)
            #预测值

    image = torch.squeeze(image, 0).permute(1, 2, 0).mul(255).numpy().astype(np.uint8)
    print(type(image))
    #转换成图片
    images = Image.fromarray(image)
    #标志为白色
    colorl = [255, 255, 255]
    count = 0
    for pred in prediction:
        for idx, mask in enumerate(pred['masks']):
            if pred['scores'][idx].item() < 0.8:       #得分小于0.6 不要
                continue

            # m = mask[0].mul(255).byte().cpu().numpy()
            #[95, 55, 674, 933] [12, 71, 688, 337]
            #获取坐标
            box = list(map(int, pred["boxes"][idx].tolist()))
            # 扩大坐标
            box = site_img(box, images)
            crop_img(box, images, idx, i)
            print(box)

            label = CATEGORIES2LABELS[pred["labels"][idx].item()]
            print(label)

            score = pred["scores"][idx].item()

            image = overlay_ann(image, box, label, score)
            print(idx)
            count += 1
            #使得图片变为白色
            #images = np.array(images)
            for col in range(box[0], box[2]):
                for row in range(box[1], box[3]):
                    image[row, col] = colorl


    # cv2.namedWindow('image')
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(r'E:\python\MaskRcnn\maskrcnn_picture\output/delete_img.png', image)
    move_img(i, count)


if __name__ == "__main__":
    import sys
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    argv = sys.argv[1:]
    print("==============argv",argv)
    # 源目录
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
        picture_main(img_path, i)
        i += 1



