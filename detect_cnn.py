#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example shows how to run a CNN based face detector using dlib.  The
#   example loads a pretrained model and uses it to find faces in images.  The
#   CNN model is much more accurate than the HOG based model shown in the
#   face_detector.py example, but takes much more computational power to
#   run, and is meant to be executed on a GPU to attain reasonable speed.
#
#   You can download the pre-trained model from:
#       http://dlib.net/files/mmod_human_face_detector.dat.bz2
#
#   The examples/faces folder contains some jpg images of people.  You can run
#   this program on them and see the detections by executing the
#   following command:
#       ./cnn_face_detector.py mmod_human_face_detector.dat ../examples/faces/*.jpg
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA
#   if you have a CPU that supports AVX instructions, you have an Nvidia GPU
#   and you have CUDA installed since this makes things run *much* faster.
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html.

import sys
import dlib
import shutil #move file
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2
import matplotlib.image as mpimg
caffe_root = '~/Caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_gpu()


detector = dlib.get_frontal_face_detector() #cpu
#detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat') #gpu & cnn
faces_folder_path = './test/'
valid_folder_path = './valid/'
trims_folder_path = './trim/'
genderpath=''
agepath=''
os.environ["GLOG_minloglevel"] = "3"
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
mean_filename = './mean.binaryproto'
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean = caffe.io.blobproto_to_array(a)[0]
age_net_pretrained = './age_net.caffemodel'
age_net_model_file = './deploy_age.prototxt'
gender_net_pretrained = './gender_net.caffemodel'
gender_net_model_file = './deploy_gender.prototxt'
age_net = caffe.Classifier(age_net_model_file, age_net_pretrained,
                           mean=mean,
                           channel_swap=(2, 1, 0),
                           raw_scale=255,
                           image_dims=(227, 227))
gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                              mean=mean,
                              channel_swap=(2, 1, 0),
                              raw_scale=255,
                              image_dims=(256, 256))
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']



for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)
    dets = detector(img, 1)

    for i, d in enumerate(dets):
        trimtop=int(d.top()-d.top()*0.2)
        trimleft= int(d.left() - d.left() * 0.2)
        trimbottom = int(d.bottom() + d.bottom() * 0.2)
        trimright = int(d.right() + d.right() * 0.2)
        img_trim=img[trimtop:trimbottom,trimleft:trimright] #detect area img cut
#        img_trim=cv2.resize(img_trim,(256,256),interpolation=cv2.INTER_CUBIC)
        img_trim=cv2.cvtColor(img_trim,cv2.COLOR_BGR2RGB) #BGR to RGB converting

        #predict
        age_prediction = age_net.predict([img_trim])
        gender_prediction = gender_net.predict([img_trim])
        print 'predicted age & gender :', age_list[age_prediction[0].argmax()], gender_list[
            gender_prediction[0].argmax()]
        

        #age,genderpath
        if(gender_list[gender_prediction[0].argmax()]=='Male'):
            genderpath='Male/'
        else:
            genderpath='Female/'
        if(age_list[age_prediction[0].argmax()] == '(0, 2)'):
            agepath='(0, 2)/'
        elif(age_list[age_prediction[0].argmax()] == '(4, 6)'):
            agepath = '(4, 6)/'
        elif (age_list[age_prediction[0].argmax()] == '(8, 12)'):
            agepath = '(8, 12)/'
        elif (age_list[age_prediction[0].argmax()] == '(15, 20)'):
            agepath = '(15, 20)/'
        elif (age_list[age_prediction[0].argmax()] == '(25, 32)'):
            agepath = '(25, 32)/'
        elif (age_list[age_prediction[0].argmax()] == '(38, 43)'):
            agepath = '(38, 43)/'
        elif (age_list[age_prediction[0].argmax()] == '(48, 53)'):
            agepath = '(48, 53)/'
        elif (age_list[age_prediction[0].argmax()] == '(60, 100)'):
            agepath = '(60, 100)/'


        #filename
        img_name_jpg = os.path.split(f)[1]
        img_name = os.path.splitext(img_name_jpg)[0]
        shutil.copy(faces_folder_path+img_name_jpg,valid_folder_path+img_name_jpg)
        #trimming face
        trimname=str(img_name)+str('-')+str(i)+str('.jpg')
        savepath=str(trims_folder_path)+str(genderpath)+str(agepath)
        trimname = os.path.join(savepath, trimname)
        #save img
        cv2.imwrite(trimname,img_trim)