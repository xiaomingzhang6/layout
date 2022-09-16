# 张晓明
# 图片二值化
from PIL import Image
import cv2

def produceImage(file_path, width, height, out_path):
    image = Image.open(file_path)
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    resized_image.save(out_path)


if __name__ == '__main__':
    import os
    project_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(project_dir, r'E:\output_title')
    # 切换目录
    os.chdir(file_path)
    # 遍历目录下所有的文件
    # 照片索引
    i = 1
    for image_name in os.listdir(os.getcwd()):
        # img = Image.open(os.path.join(input, image_name))
        img_path = os.path.join(file_path, image_name)
        print(img_path)
        img = cv2.imread(img_path, 1)
        # 对图片做灰度化转换。
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 再进行图片标准化,将图片数组的数值统一到一定范围内。函数的参数
        # 依次是：输入数组，输出数组，最小值，最大值，标准化模式。
        h = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite( r'E:\output_title\result{}.png'.format(i), h)
        i += 1
    # # 从文件路径中读入图片。
    # file_path = r'E:\output_title\output2.png'
    # #输出图片
    # out_path = r'E:\output_title\result1.png'
    # img = cv2.imread(file_path, 1)
    # # 对图片做灰度化转换。
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # 再进行图片标准化,将图片数组的数值统一到一定范围内。函数的参数
    # # 依次是：输入数组，输出数组，最小值，最大值，标准化模式。
    # h = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    # cv2.imwrite(out_path, h)
    # # #改变像素值
    # # width = 800
    # # height = 700
    # # produceImage(file_path, width, height, out_path)