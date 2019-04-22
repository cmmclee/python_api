# -*-coding:utf-8-*-
from keras.preprocessing.image import img_to_array
import cv2
from keras.models import load_model
import numpy as np
from flask import Flask, request
import urllib
import time

app = Flask(__name__)

detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
# 分类器和模型初始化
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

def detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) > 0:
        if len(faces) > 1:
            faces = sorted(faces, reverse=True, key=lambda x: x[2] * x[3])
        (fX, fY, fW, fH) = faces[0]
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        smile = preds[3] * 100
        label = "smile value: {:.2f}".format(smile)
        print(label)
        return label

# 注意：模型初始化后，一定要立即执行一次model.predict()，否则会报错
init_pic = cv2.imread('smile0.jpg')
detect(init_pic)

@app.route('/')
def hello():
    return "hello"

# URL到图片:下载图片--> Numpy array --> opencv格式
def url_to_image(url):
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

@app.route('/predict', methods=['GET'])
def predict():
    # userid = request.args.get('userid')
    pickey = request.args.get('pickey')
    starttime = time.time()

    ossUrl = 'http://longbei-dev-media-out.oss-cn-beijing.aliyuncs.com/'
    imgUrl = ossUrl + pickey
    image = url_to_image(imgUrl)

    loadtime = time.time()
    print("load pic take seconds: {:f}".format(loadtime - starttime))
    # 尝试检测不同的图片 # img = "smile{}.jpg".format()
    # 尝试检测不同的情景(没有人脸的，多张人脸的)
    # 性能测试、并发测试
    # 部署到测试环境
    # 输入格式的兼容与优化
    # 与face++ api 的比较
    label = detect(image)

    endtime = time.time()
    print("all steps take seconds: {:f}".format(endtime - starttime))
    return label

def test():
    predict('classroom/53e75ca5-aac8-4f2a-bf35-a6ced65c8248')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, use_reloader=False)
    # test()


