import cv2
import numpy as np
import exifread
from matplotlib import pyplot as plt

img_path = 'images/1/iso100ss100.jpg'

def grayscale_hist(img_path):
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('iso100ss100',gray_img)

    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('iso100ss100',gray_img)
    hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
    plt.hist(gray_img.ravel(),256,[0,256])
    plt.title('Histogram for gray scale picture')
    plt.show()

    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 27: break             # ESC key to exit
    cv2.destroyAllWindows()

def numpy_hist(img_path):
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('iso100ss100', gray_img)
    # hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
    hist, bins = np.histogram(gray_img, 256, [0, 256])

    fig = plt.figure()
    fig.canvas.set_window_title('Gray Scale Picture with numpy')
    fig.suptitle('iso100ss100', fontsize=10)
    plt.hist(gray_img.ravel(), 256, [0, 256])

    plt.show()

    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 27: break  # ESC key to exit
    cv2.destroyAllWindows()

def color_hist(img_path):
    img = cv2.imread(img_path, -1)
    cv2.imshow('iso100ss100', img)

    fig = plt.figure()
    fig.canvas.set_window_title('Color Histogram')
    fig.suptitle('iso100ss100', fontsize=10)

    color = ('b', 'g', 'r')
    for channel, col in enumerate(color):
        histr = cv2.calcHist([img], [channel], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    plt.show()

    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 27: break  # ESC key to exit
    cv2.destroyAllWindows()

def myreadexif(img_path):

    exif = open(img_path, 'rb')

    tags = exifread.process_file(exif)

    for tag in tags.keys():
        if tag not in ('Image','JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote','Interoperability'):
            print(tag, tags[tag])

def pixaverage()
    with picamera.PiCamera() as camera:
        camera.resolution = (100, 75)
        with picamera.array.PiRGBArray(camera) as stream:
            camera.exposure_mode = 'auto'
            camera.awb_mode = 'auto'
            camera.capture(stream, format='rgb')
            pixAverage = int(np.average(stream.array[..., 1]))
    print("Light Meter pixAverage=%i" % pixAverage)

if __name__ == '__main__':

    try:
        #grayscale_hist(img_path)
        #numpy_hist(img_path)
        #color_hist(img_path)
        myreadexif(img_path)

    except OSError  as e:
        print("Error: " + str(e))