# pykinect2 for KinectV2
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

#for insta360 mask-rcnn
# Mask RCNN Based on
import pickle
import tensorflow as tf
from tensorflow import keras
import skimage
from ModelConfig import *
import logging

# Mrcnn folder to import MRcnn config
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib

from flask import Response, Flask, render_template

import os
import numpy as np
import cv2
import torch
from util import rgb2hsl
import sys
sys.path.insert(0, "..")
from datetime import datetime

from opcua import Client
from opcua import ua

app = Flask(__name__)

# from IPython import embed
# logging.basicConfig(level=logging.DEBUG)
client = Client("opc.tcp://192.168.219.15:4840/")
# client = Client("opc.tcp://olivier:olivierpass@localhost:53530/OPCUA/SimulationServer/")
# client.set_security_string("Basic256Sha256,SignAndEncrypt,certificate-example.der,private-key-example.pem")

client.connect()
root = client.get_root_node()

print("Root is", root)
print("childs of root are: ", root.get_children())
print("name of root is", root.get_browse_name())
objects = client.get_objects_node()
print("childs og objects are: ", objects.get_children())
myfloat = client.get_node("ns=4;s=Float")
mydouble = client.get_node("ns=4;s=Double")
myint64 = client.get_node("ns=4;s=Int64")
myuint64 = client.get_node("ns=4;s=UInt64")
myint32 = client.get_node("ns=4;s=Int32")
myuint32 = client.get_node("ns=4;s=UInt32")

class SubHandler(object):

    """
    Client to subscription. It will receive events from server
    """

    def datachange_notification(self, node, val, data):
        print("Python: New data change event", node, val)

    def event_notification(self, event):
        print("Python: New event", event)

handler = SubHandler()
sub = client.create_subscription(1000, handler)

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

colors = random_colors(1)
# for KinectV2 Model - Leaf Detection
model_path = 'pretrained_weight/happytree1.pt'
model_kinect = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

#for Insta360 Model
ROOT_DIR = 'C:/Users/CS-DE-09/PycharmProjects/videostreamweb/'
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
CUSTOM_MODELPTH = os.path.join(ROOT_DIR, "pretrained_model/mask_rcnn_bead_and_defect_0008.h5")
frame_idx = 0
config = inference_config
config.print()

model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
model.load_weights(CUSTOM_MODELPTH, by_name=True)
class_names = [
    'BG', 'Bead', 'Spatter', 'Porosity', 'Crack'
]

# camera = cv2.VideoCapture('rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen03.stream')  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
kinectD = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
insta360 = cv2.VideoCapture(1)

#check field device status. default is false
sensorinfrared = 3
sensorcolor = 3

def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

def display_instances(image, boxes, masks, ids, names, scores,frame_idx):
    """
        take the image and results and apply the mask, box, and Label
    """

    #counting defect weld
    c_spatter = 0
    c_porosity = 0
    c_crack = 0
    c_bead = 0

    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)
    frame_idx += 1
    if not n_instances:
         print('NO INSTANCES TO DISPLAY')
    else:
         assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    #
    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        if label == 'Spatter':
            c_spatter +=1
        elif label == 'Porosity':
            c_porosity +=1
        elif label =="Crack":
            c_crack +=1
        else:
            c_bead += 1

        # ObjectPosition Variable
        xminvar = client.get_node('ns=3;s="defect_x_min"')
        xminval = x1
        xminvar.set_value(ua.DataValue(ua.Variant(xminval, ua.VariantType.Float)))

        xmaxvar = client.get_node('ns=3;s="defect_x_max"')
        xmaxval = x2
        xmaxvar.set_value(ua.DataValue(ua.Variant(xmaxval, ua.VariantType.Float)))

        yminvar = client.get_node('ns=3;s="defect_y_min"')
        ymixval = y1
        yminvar.set_value(ua.DataValue(ua.Variant(ymixval, ua.VariantType.Float)))

        ymaxvar = client.get_node('ns=3;s="defect_y_max"')
        ymaxval = y2
        ymaxvar.set_value(ua.DataValue(ua.Variant(ymaxval, ua.VariantType.Float)))

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2
        )

    # ObjectColorRGB variable
    n_spatter = client.get_node('ns=3;s="defect_spatter"')
    v_spatter = c_spatter
    n_spatter.set_value(ua.DataValue(ua.Variant(v_spatter, ua.VariantType.Int32)))

    n_porosity = client.get_node('ns=3;s="defect_porosity"')
    v_porosity = c_porosity
    n_porosity.set_value(ua.DataValue(ua.Variant(v_porosity, ua.VariantType.Int32)))

    n_crack = client.get_node('ns=3;s="defect_crack"')
    v_crack = c_crack
    n_crack.set_value(ua.DataValue(ua.Variant(v_crack, ua.VariantType.Int32)))

    return image

def gen_frames_kinect():  # generate frame by frame from camera
    while True:
        detected_number = 0
        # Capture frame-by-frame
        frame = kinect.get_last_color_frame()  # read the camera frame
        sensorcolor = 1
        frame = np.reshape(frame, (1080, 1920, 4))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        # init Depthframe
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frameD = kinectD.get_last_depth_frame()
        sensorinfrared = 1
        frameDepth = kinectD._depth_frame_data
        frameD = frameD.astype(np.uint8)
        frameD = np.reshape(frameD, (424, 512))
        frameD = cv2.cvtColor(frameD, cv2.COLOR_GRAY2BGR)
        # pretrained model to express arrays without indexes
        results = model_kinect(frame, size=640)
        # variable box shows DataFrame as result.
        box = results.pandas().xyxy[0][:5]  # among many objects, picked 5 objects from the highest confidence
        box = box.sort_values(["xmin"], ascending=(
            True))  # And give the number from the left using Xmin which defines number from the left
        # each array saves into one array
        confidence = box['confidence']
        xmin = box['xmin']
        xmax = box['xmax']
        ymin = box['ymin']
        ymax = box['ymax']
        label = box['name']

        # Zipped data to express realtime detected
        for color, xmini, xmaxi, ymini, ymaxi, labelname, conf in zip(colors, xmin, xmax, ymin, ymax, label,
                                                                      confidence):
            # for Distance from camera to object, pinpoint the middle number to Depth data
            minpoint = (int(xmini), int(ymini))
            maxpoint = (int(xmaxi), int(ymaxi))
            depthmmshow = (int(xmini), int(ymaxi))
            x_center = int((((xmini) + (xmaxi)) / 2))
            y_center = int((((ymini) + (ymaxi)) / 2))
            Center = (int(x_center / 2), int(y_center * .8))
            # print(x_center,y_center,Center)
            Pixel_Depth = frameDepth[((int(y_center * .8) * 512) + int(x_center / 2))]
            # print(Pixel_Depth)

            number_index = 1

            # realtime Object size
            # two coordinates of each x and y are shown. subtract each max coordinate to min coordinate to find horizontal and vertical
            xpxsize = int(xmaxi) - int(xmini)
            ypxsize = int(ymaxi) - int(ymini)
            Distanceformm = Pixel_Depth / 1000
            actualx = xpxsize * Distanceformm
            actualy = ypxsize * Distanceformm

            # for single object detection to detect color, using cropping
            frameobjectcolor = frame[int(ymini):int(ymaxi), int(xmini):int(xmaxi)]

            # Record current date into save images without confusion
            currentdate = datetime.now()
            current_datetime = currentdate.strftime("%Y-%m-%d %H.%M")

            # find the object color to remove background subtraction
            # use the average with mask is simpler than using contours or transparent
            frameHSV = cv2.cvtColor(frameobjectcolor, cv2.COLOR_RGB2HSV)
            lowercolor = (0, 50, 50)
            highercolor = (359, 255, 255)
            frame_mask = cv2.inRange(frameHSV, lowercolor, highercolor)
            framemasked = cv2.bitwise_and(frameobjectcolor, frameobjectcolor, mask=frame_mask)
            framereshape = framemasked.reshape((framemasked.shape[0] * framemasked.shape[1], 3))
            average_color = cv2.mean(frameobjectcolor, mask=frame_mask)

            # gather color attribute from average color
            b, g, r = average_color[:3]
            r = round(r)
            g = round(g)
            b = round(b)
            # print(r,g,b)
            # convert to 0~1 range to convert HSL by diving 255
            rl = r / 255
            gl = g / 255
            bl = b / 255
            # Converting RGB to HSL is defined in util.py file
            h, s, l = rgb2hsl(rl, gl, bl)
            h = round(h * 60)
            s = round(s * 100)
            l = round(l * 100)
            # print(h,s,l)

            # define variable to plot object name above the box
            label = labelname

            # Drawing Rectangle as bounding box from predefined coordinates
            frame = cv2.rectangle(frame, (minpoint[0], minpoint[1]), (maxpoint[0], maxpoint[1]), (0, 0, 0), 1)
            frame = cv2.rectangle(frame, (minpoint[0], minpoint[1]), (minpoint[0] + 10, minpoint[1] + 10), (b, g, r),
                                  -1)
            # plotting square as colorbox to show color
            frameD = cv2.circle(frameD, Center, 10, color, -1)

            # Attribute for plotting size
            text = '{}{} Depth: {}mm'.format(label, detected_number, Pixel_Depth)
            textD = 'x:{:.2f}mm, y:{:.2f}mm'.format(actualx, actualy)
            frame = cv2.putText(frame, text, minpoint, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
            frame = cv2.putText(frame, textD, depthmmshow, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)


            # Variable Group from Field Device
            # ObjectName node Variable
            Namevar = client.get_node('ns=3;s="obj_name_index"')
            Nameval = number_index
            Namevar.set_value(ua.DataValue(ua.Variant(Nameval, ua.VariantType.Int32)))

            # ObjectPosition Variable
            xminvar = client.get_node('ns=3;s="Pos_X_min"')
            xminval = xmini
            xminvar.set_value(ua.DataValue(ua.Variant(xminval, ua.VariantType.Float)))

            xmaxvar = client.get_node('ns=3;s="Pos_X_max"')
            xmaxval = xmaxi
            xmaxvar.set_value(ua.DataValue(ua.Variant(xmaxval, ua.VariantType.Float)))

            yminvar = client.get_node('ns=3;s="Pos_Y_min"')
            ymixval = ymini
            yminvar.set_value(ua.DataValue(ua.Variant(ymixval, ua.VariantType.Float)))

            ymaxvar = client.get_node('ns=3;s="Pos_Y_max"')
            ymaxval = ymaxi
            ymaxvar.set_value(ua.DataValue(ua.Variant(ymaxval, ua.VariantType.Float)))

            # ObjectColorRGB variable
            redvar = client.get_node('ns=3;s="Color_R"')
            redval = r
            redvar.set_value(ua.DataValue(ua.Variant(redval, ua.VariantType.Float)))

            greenvar = client.get_node('ns=3;s="Color_G"')
            greenval = g
            greenvar.set_value(ua.DataValue(ua.Variant(greenval, ua.VariantType.Float)))

            bluevar = client.get_node('ns=3;s="Color_B"')
            blueval = b
            bluevar.set_value(ua.DataValue(ua.Variant(blueval, ua.VariantType.Float)))

            # ObjectColorHSL Variable
            huevar = client.get_node('ns=3;s="Color_H"')
            hueval = h
            huevar.set_value(ua.DataValue(ua.Variant(hueval, ua.VariantType.Float)))

            satvar = client.get_node('ns=3;s="Color_S"')
            satval = s
            satvar.set_value(ua.DataValue(ua.Variant(satval, ua.VariantType.Float)))

            lgtvar = client.get_node('ns=3;s="Color_L"')
            lgtval = l
            lgtvar.set_value(ua.DataValue(ua.Variant(lgtval, ua.VariantType.Float)))

            # ObjectColorSize Variable
            depvar = client.get_node('ns=3;s="Size_Depth"')
            depval = Distanceformm
            depvar.set_value(ua.DataValue(ua.Variant(depval, ua.VariantType.Float)))

            horvar = client.get_node('ns=3;s="Size_horizontal"')
            horval = actualx
            horvar.set_value(ua.DataValue(ua.Variant(horval, ua.VariantType.Float)))

            vervar = client.get_node('ns=3;s="Size_vertical"')
            verval = actualy
            vervar.set_value(ua.DataValue(ua.Variant(verval, ua.VariantType.Float)))

            # # MachineStatus Variable
            # scolorvar = client.get_node("ns=3;i=1022")
            # scolorval = sensorcolor
            # scolorvar.set_value(ua.Variant(scolorval, ua.VariantType.Int32))

            # sinfrvar = client.get_node("ns=3;i=1019")
            # sinfrval = sensorinfrared
            # sinfrvar.set_value(ua.Variant(sinfrval, ua.VariantType.Int32))

            number_index += 1
            detected_number += 1

        ret, buffer = cv2.imencode('.jpg', frame)
        # if not ret:
        #     print("No Instance to display")
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed_kinect')
def video_feed_kinect():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames_kinect(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = insta360.read()  # read the camera frame
        if not success:
            break
        else:
            colorActive = 1
            # MachineStatus Variable
            # scolorvar = client.get_node("ns=8;i=1038")
            # scolorval = colorActive
            # scolorvar.set_value(ua.Variant(scolorval, ua.VariantType.Int32))
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("No Instance to display")
            results = model.detect([frame])
            r = results[0]

            masked_frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'],
                                                 frame_idx)
            masked_frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + masked_frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True, threaded=True)