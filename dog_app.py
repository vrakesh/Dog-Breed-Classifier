from uuid import uuid4
import os
from flask import Flask, request, render_template, send_from_directory
from face_detector import face_detect
from dog_detector import df
import numpy as np
import cv2
from glob import glob

app = Flask(__name__, template_folder='.')
# app = Flask(__name__, static_folder="images")



APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    destination = ''
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)

    input_file = np.array(glob(destination))[0]
    hello_str = 'hello'
    look_str = 'Your matching dog breed is '
    dog_str  = 'The dog is a '
    second_str = ''
    print("Reading ",input_file)
    #Human Detected
    if(face_detect(input_file) ):
        hello_str = ''.join([hello_str, ' human'])
        second_str = look_str
    #Dog detected
    elif(df.dog_detector(input_file)):
        second_str = dog_str
    #Wolf is neither a human or dog
    else:
        print("Could not find dog or human")
    df.dog_model()
    df.load_weights()
    output = df.Resnet50_predict_breed(input_file)
    second_str = ''.join([second_str,output])
    print(second_str)
    return render_template("complete.html", image_name=filename, first_str=hello_str, second_str=second_str)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(port=8080, debug=True)
