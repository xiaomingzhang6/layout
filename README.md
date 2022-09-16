# layout
一．训练模型
1.训练表格模型--maskrcnn_table
打开maskrcnn_table/T/train.py  
把数据集放入maskrcnn_table/T/PennFudanPed/Images，mask放入maskrcnn_table/T/PennFudanPed/Mask，运行train.py即可

2.训练图片模型--maskrcnn_picture
打开maskrcnn_picture/T/train.py  
把数据集放入maskrcnn_picture/T/PennFudanPed/Images，mask放入maskrcnn_picture/T/PennFudanPed/Mask，运行train.py即可
3.训练标题模型--maskrcnn_title
打开maskrcnn_title/T/train.py  把数据集放入maskrcnn_title/T/PennFudanPed/Images，mask放入maskrcnn_title/T/PennFudanPed/Mask，运行train.py即可

二．提取表格
首先要提取表格，打开maskrcnn_table/infer.py。需要改变的就是路径，找到main中的input：要提取表格的图像放入该文件夹中。
crop_table()函数是提取表格的函数，path是保存表格的路径。
move_img()函数是把输出的表格复制到文件夹里，dst_path是保存切割表格的目的路径。output_path是保存输出照片的目的路径。dst_output_path是保存的根路径。src_path同path

三．提取图片
其次提取图片，打开maskrcnn_picture/infer.py。需要改变的就是路径，找到main中的input：要提取图片的图像放入该文件夹中，也就是刚提取完表格的图片放入里面。
crop_img()是提取图片的函数，path是保存图片的路径。
move_img()函数是把输出的图片复制到文件夹里，dst_path是保存切割照片的目的路径。output_path是保存输出照片的目的路径。dst_output_path是保存的根路径。src_path同path

四．红章去除
打开image enhancement/remove_red_stamp.py

五．标注标题
最后标注标题，打开maskrcnn_title/infer.py。需要改变的就是路径，找到main中的input：要提取图片的图像放入该文件夹中，也就是刚去除完红章的照片放入里面。
Crop()函数是将一页一分为二
move_img()函数是把把框选的标题复制到文件夹里，output_path是保存输出框选的标题的目的路径。dst_output_path是保存的根路径

六．图像增强
打开image enhancement/eval_illumination.py。需要改变的就是路径，找到main中的input：图像放入该文件夹中
