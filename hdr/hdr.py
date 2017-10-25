#!/usr/bin/env python
#Quelle: http://www.learnopencv.com/high-dynamic-range-hdr-imaging-using-opencv-cpp-python/

import cv2
import numpy as np
from os import listdir, makedirs, walk,path
from os.path import isfile, join

def readImagesAndTimes(picturepath):
  
  #times = np.array([ 1/30.0, 0.25, 2.5, 15.0 ], dtype=np.float32)
  #filenames = ["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]


  filenames = []
  shuttertimes = []

  onlyfiles = [f for f in listdir(picturepath) if isfile(join(picturepath, f))]

  for file in onlyfiles:
    print(file)
    filenames.append(picturepath+file)
    shuttertimes.append(np.float32(file.split("ss")[1].replace('.jpg',''))/1000000)



  #filenames = [picturepath +"iso100ss200.jpg",picturepath +"iso100ss600.jpg",picturepath +"iso100ss4000.jpg"]
  #s1 = np.float32(filenames[0].split("ss")[1].replace('.jpg',''))/1000000
  #s2 = np.float32(filenames[1].split("ss")[1].replace('.jpg',''))/1000000
  #s3 = np.float32(filenames[2].split("ss")[1].replace('.jpg',''))/1000000
  #print("Shutter time s1 : "+str(s1))
  #print("Shutter time s2 : "+str(s2))
  #print("Shutter time s3 : "+str(s3))

  times = np.array(shuttertimes, dtype=np.float32)   #times = np.array([ 0.0001, 0.0006, 0.002 ], dtype=np.float32)

  images = []
  for filename in filenames:
    im = cv2.imread(filename)
    images.append(im)
  
  return images, times


def makeHDR (picturepath,outputfolder):

  try:
    images, times = readImagesAndTimes(picturepath)

    # Align input images
    print("Aligning images ... ")
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)

    # Obtain Camera Response Function (CRF)
    print("Calculating Camera Response Function (CRF) ... ")
    calibrateDebevec = cv2.createCalibrateDebevec()
    responseDebevec = calibrateDebevec.process(images, times)

    # Merge images into an HDR linear image
    print("Merging images into one HDR image ... ")
    mergeDebevec = cv2.createMergeDebevec()
    hdrDebevec = mergeDebevec.process(images, times, responseDebevec)
    # Save HDR image.
    cv2.imwrite(join(outputfolder,"hdrDebevec.hdr"), hdrDebevec)
    print("saved hdrDebevec.hdr ")

    # Tonemap using Drago's method to obtain 24-bit color image
    print("Tonemaping using Drago's method ... ")
    tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
    ldrDrago = tonemapDrago.process(hdrDebevec)
    ldrDrago = 3 * ldrDrago
    cv2.imwrite(join(outputfolder,"ldr-Drago.jpg"), ldrDrago * 255)
    print("saved ldr-Drago.jpg")

    # Tonemap using Durand's method obtain 24-bit color image
    print("Tonemaping using Durand's method ... ")
    tonemapDurand = cv2.createTonemapDurand(1.5, 4, 1.0, 1, 1)
    ldrDurand = tonemapDurand.process(hdrDebevec)
    ldrDurand = 3 * ldrDurand
    cv2.imwrite(join(outputfolder,"ldr-Durand.jpg"), ldrDurand * 255)
    print("saved ldr-Durand.jpg")

    # Tonemap using Reinhard's method to obtain 24-bit color image
    print("Tonemaping using Reinhard's method ... ")
    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
    ldrReinhard = tonemapReinhard.process(hdrDebevec)
    cv2.imwrite(join(outputfolder,"ldr-Reinhard.jpg"), ldrReinhard * 255)
    print("saved ldr-Reinhard.jpg")

    # Tonemap using Mantiuk's method to obtain 24-bit color image
    print("Tonemaping using Mantiuk's method ... ")
    tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
    ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
    ldrMantiuk = 3 * ldrMantiuk
    cv2.imwrite(join(outputfolder,"ldr-Mantiuk.jpg"), ldrMantiuk * 255)
    print("saved ldr-Mantiuk.jpg")

  except cv2.error as e:
    print("Error: " + str(e))

if __name__ == '__main__':

  subdirs = []
  foldername = './pictures_forhdr/'

  # Read images and exposure times
  print("Reading images ... ")
  try:

    for i, j, y in walk(foldername):
      subdirs.append(i)
    subdirs.pop(0)

    for picturesdirectory in subdirs:
      newfoldername = str(picturesdirectory).split("normal/")[1]
      outputfolder = join('./output/', ('hdr_' + newfoldername + "/"))
      print(outputfolder)
      if not path.exists(outputfolder):
        makedirs(outputfolder)
      print(picturesdirectory + str('/'))
      makeHDR(picturesdirectory+'/', outputfolder)


  except OSError  as e:
    print("Error: " + str(e))



