import numpy as np
import sys, os, imp
import cv2
import transformations
import features
import traceback

from PIL import Image

import pdb

def helper():
    HKD = features.HarrisKeypointDetector()
    SFD = features.SimpleFeatureDescriptor()
    MFD = features.MOPSFeatureDescriptor()
    image = np.array(Image.open('resources/triangle1.jpg'))
    grayImage = cv2.cvtColor(image.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)
    (a,b) = HKD.computeHarrisValues(grayImage) # Todo1
    c = HKD.computeLocalMaxima(a) # Todo2
    d = HKD.detectKeypoints(image) # Todo3
    e = SFD.describeFeatures(image, d) # Todo 4
    f = MFD.describeFeatures(image, d) # Todo 5,6
    return f 

def testSame():
    HKD = features.HarrisKeypointDetector()
    SFD = features.SimpleFeatureDescriptor()
    MFD = features.MOPSFeatureDescriptor()
    image = np.array(Image.open('resources/triangle1.jpg'))
    grayImage = cv2.cvtColor(image.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)
    (a,b) = HKD.computeHarrisValues(grayImage) # Todo1
    c = HKD.computeLocalMaxima(a) # Todo2
    d = HKD.detectKeypoints(image) # Todo3
    e = SFD.describeFeatures(image, d) # Todo 4
    f = MFD.describeFeatures(image, d) # Todo 5,6

    loaded = np.load('resources/arrays.npz', allow_pickle=True)
    correct = loaded['f']

    cnonZero = np.nonzero(correct)[0]
    tnonZero = np.nonzero(f)[0]

    indices = np.intersect1d(cnonZero, tnonZero)
    sum=0

    for i in range(len(indices)):
        if not np.allclose(correct[indices[i]],f[indices[i]],rtol=1e-3,atol=1e-3):
            sum+=1
    return sum

def testDifferent():
    HKD = features.HarrisKeypointDetector()
    SFD = features.SimpleFeatureDescriptor()
    MFD = features.MOPSFeatureDescriptor()
    image = np.array(Image.open('resources/triangle1.jpg'))
    grayImage = cv2.cvtColor(image.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)
    (a,b) = HKD.computeHarrisValues(grayImage) # Todo1
    c = HKD.computeLocalMaxima(a) # Todo2
    d = HKD.detectKeypoints(image) # Todo3
    e = SFD.describeFeatures(image, d) # Todo 4
    f = MFD.describeFeatures(image, d) # Todo 5,6

    loaded = np.load('resources/arrays.npz', allow_pickle=True)
    correct = loaded['f']

    cnonZero = np.nonzero(correct)[0]
    tnonZero = np.nonzero(f)[0]

    indices = np.setdiff1d(tnonZero, cnonZero)
    sum=0

    for i in range(len(indices)):
        if not np.allclose(correct[indices[i]],f[indices[i]],rtol=1e-3,atol=1e-3):
            print(indices[i])
    return sum

def testMatch():
    HKD = features.HarrisKeypointDetector()
    SFD = features.SimpleFeatureDescriptor()
    MFD = features.MOPSFeatureDescriptor()
    SFM = features.SSDFeatureMatcher()
    image = np.array(Image.open('resources/triangle1.jpg'))
    grayImage = cv2.cvtColor(image.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)
    (a,b) = HKD.computeHarrisValues(grayImage) # Todo1
    c = HKD.computeLocalMaxima(a) # Todo2
    d = HKD.detectKeypoints(image) # Todo3
    e = SFD.describeFeatures(image, d) # Todo 4
    f = MFD.describeFeatures(image, d) # Todo 5,6
    return SFM.matchFeatures(f,f)



