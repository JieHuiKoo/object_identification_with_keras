#! /home/jiehui/anaconda3/envs/tensorflow/bin/python

import rospy
import sys
import numpy as np
import tensorflow as tf # Must be imported first before cv_bridge if not will throw up segmentation fault error
from tensorflow import keras
import cv2
import rospkg

from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Float32

print("Python Version: " + str(sys.version_info[0]) + '.' + str(sys.version_info[1]))
print("OpenCV Version: " + str(cv2.__version__))

image_size = (180, 180)
rospack = rospkg.RosPack()
cwd = rospack.get_path('object_identification_with_keras')
model = keras.models.load_model(cwd + "/src/nodes/SavedModel")

def imgmsg_to_cv2(img_msg):
    
    rgb8_flag = 0
    if img_msg.encoding != "bgr8":
        rgb8_flag = 1
   
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                    dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()

    if rgb8_flag:
        image_opencv = cv2.cvtColor(image_opencv, cv2.COLOR_RGB2BGR)
       
    return image_opencv

def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg

def convert_to_keras_array(img):
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    return img_array

def extract_prediction(predictions):
    class_array = ['drink', 'pot', 'trash']
    
    predicted_class_num = np.argmax(predictions)
    
    return class_array[predicted_class_num], predictions[0][predicted_class_num]

def make_prediction(image, model):
    predictions = model.predict(convert_to_keras_array(image), verbose=0)

    return extract_prediction(predictions)

def process_image(image_msg):

    # Declare the cvBridge object
    proc_image = imgmsg_to_cv2(image_msg)
    proc_image = cv2.resize(proc_image, image_size, interpolation = cv2.INTER_AREA)
    object_class, confidence = make_prediction(proc_image, model)

    object_class_pub = rospy.Publisher('armCamera/nearestObject_Class', String, queue_size=1)
    object_class_pub.publish(object_class)

    object_class_pub = rospy.Publisher('armCamera/nearestObject_Confidence', Float32, queue_size=1)
    object_class_pub.publish(confidence)

    # print(object_class, confidence)
    
def start_node():
    rospy.init_node('image_masking')
    rospy.loginfo('image_identify node started')

    rospy.Subscriber("armCamera/nearestObject_Image", Image, process_image)
    
    rospy.spin()

if __name__ == '__main__':
    
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass

