# 张晓明
from flask import Flask, jsonify, abort, request, render_template, redirect

import os
import sys

import maskrcnn_picture,maskrcnn_title,maskrcnn_table,image_enhancement

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = ['maskrcnn_picture', 'maskrcn_table', 'maskrcnn_title','image_enhancement']

for name in PATHS:
    sys.path.append(os.path.join(ROOT_DIR, name))
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
from PIL import Image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/picture_pretrain/',methods = ['POST', 'GET'])
def picture_train():
    if request.method == 'POST':
        # my_iter = request.form['my_iter']
        # my_batch_size = request.form['my_batch_size']
        # my_iter = int(my_iter)
        # my_batch_size = int(my_batch_size)
        putlist = maskrcnn_picture.T.train.main()
        return render_template('picture_train.html', aa=putlist)
    else:
        return render_template('picture_train')
@app.route('/table_pretrain/',methods = ['POST', 'GET'])
def table_train():
    if request.method == 'POST':
        # my_iter = request.form['my_iter']
        # my_batch_size = request.form['my_batch_size']
        # my_iter = int(my_iter)
        # my_batch_size = int(my_batch_size)
        putlist = maskrcnn_table.T.train.main()
        return render_template('table_train', aa=putlist)
    else:
        return render_template('table_train')
@app.route('/title_pretrain/',methods = ['POST', 'GET'])
def title_train():
    if request.method == 'POST':
        # my_iter = request.form['my_iter']
        # my_batch_size = request.form['my_batch_size']
        # my_iter = int(my_iter)
        # my_batch_size = int(my_batch_size)
        putlist = maskrcnn_title.T.train.main()
        return render_template('title_train', aa=putlist)
    else:
        return render_template('title_train')
@app.route('/picture_test/',methods = ['POST', 'GET'])
def flask_picture_test():
   if request.method == 'POST':
        path = request.form['path']
        print(type(path))
        maskrcnn_picture.infer.main(path=path)
        return render_template('picture_test.html',aa='运行完毕')
   else:
       return render_template('picture_test.html')
@app.route('/table_test/',methods = ['POST', 'GET'])
def flask_table_test():
   if request.method == 'POST':
        path = request.form['path']
        print(type(path))
        maskrcnn_table.infer.main(path=path)
        return render_template('table_test.html',aa='运行完毕')
   else:
       return render_template('table_test.html')
@app.route('/title_test/',methods = ['POST', 'GET'])
def flask_title_test():
   if request.method == 'POST':
        path = request.form['path']
        print(type(path))
        maskrcnn_title.infer.main(path=path)
        return render_template('title_test.html',aa='运行完毕')
   else:
       return render_template('title_test.html')
@app.route('/enhancement_test/',methods = ['POST', 'GET'])
def flask_image_enhancement_test():
   if request.method == 'POST':
        path = request.form['path']

        image_enhancement.eval_illumination.main(path=path)

        return render_template('image_enhancement_test.html',aa='运行完毕')
   else:
       return render_template('image_enhancement_test.html')
@app.route('/enhancement_test/',methods = ['POST', 'GET'])
def remove_red_stamp():
   if request.method == 'POST':
        input_path = request.form['input_path']
        output_path = request.form['output_path']

        image_enhancement.remove_red_stamp.main(input_path=input_path,output_path=output_path)

        return render_template('remove_red_stamp.html',aa='运行完毕')
   else:
       return render_template('remove_red_stamp.html')

if __name__ == '__main__':
    app.run(debug=True)