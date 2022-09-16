'''
----张晓明----2022.7---
----去除红章----
'''
import cv2
import numpy as np
import os
from change_number import change_number
class SealRemove(object):
    """
    印章处理类
    """
    def remove_red_seal(self, image):
        """
        去除红色印章
        """
        # 获得红色通道
        blue_c, green_c, red_c = cv2.split(image)

        # 多传入一个参数cv2.THRESH_OTSU，并且把阈值thresh设为0，算法会找到最优阈值
        thresh, ret = cv2.threshold(red_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 实测调整为95%效果好一些
        filter_condition = int(thresh * 0.95)

        _, red_thresh = cv2.threshold(red_c, filter_condition, 255, cv2.THRESH_BINARY)

        # 把图片转回 3 通道
        result_img = np.expand_dims(red_thresh, axis=2)
        result_img = np.concatenate((result_img, result_img, result_img), axis=-1)

        return result_img


if __name__ == '__main__':
    input_path = r'E:\python\img\output_img'
    output_path = r'E:\python\img\delete_red'
    # change_number(input_path)
    # 源目录
    project_dir = os.path.dirname(os.path.abspath(__file__))
    input = os.path.join(project_dir, input_path)
    # 切换目录
    os.chdir(input)
    # 遍历目录下所有的文件
    #照片索引
    i = 1
    for image_name in os.listdir(os.getcwd()):
        #img = Image.open(os.path.join(input, image_name))
        img_path = os.path.join(input, image_name)
        print(img_path)
        img = cv2.imread(img_path)
        seal_rm = SealRemove()
        rm_img = seal_rm.remove_red_seal(img)
        cv2.imwrite(output_path+'\\'+(str(i).zfill(7))+'.png', rm_img)
        i += 1


