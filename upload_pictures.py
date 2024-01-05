# coding:utf-8
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from PIL import Image
import time
import cv2
from datetime import timedelta

from deeplab import DeeplabV3

app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'bmp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


deeplab_model = DeeplabV3()
name_classes = ["_background_", "smoke"]


@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "图片格式不支持"})

        basepath = os.path.dirname(__file__)

        filename = secure_filename(f.filename)
        upload_path = os.path.join(basepath, 'static/images', filename)

        f.save(upload_path)

        img = cv2.imread(upload_path)
        cv2_path = os.path.join(basepath, 'static/images', 'test.jpg')
        cv2.imwrite(cv2_path, img)

        image = Image.open(cv2_path)
        r_image = deeplab_model.detect_image(image, count=False, name_classes=name_classes)
        output_filename = 'test.jpg'
        output_path = os.path.join(basepath, 'static/results', output_filename)

        if not os.path.exists(os.path.join(basepath, 'static/results')):
            os.makedirs(os.path.join(basepath, 'static/results'))

        r_image.save(output_path)

        full_url = request.url_root + output_path.lstrip('.')

        return render_template('upload_ok.html', val1=time.time(), filename=output_filename, display_image=full_url)

    return render_template('upload.html')


@app.route('/display/<filename>')
def display_image(filename):
    return send_from_directory(app.static_folder, 'results/' + filename, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
