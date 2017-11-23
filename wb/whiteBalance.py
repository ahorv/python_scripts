import cv2
import math
import numpy as np
from PIL import Image


global images_in
#images_in = r'C:\Hoa_Python_Projects\python_scripts\hdr\input'
images_in = r'C:\Hoa_Python_Projects\python_scripts\hdr\input\20171025_140139'

class WhiteBalance:
    color = ('b', 'g', 'r')

    def gray_world(self,nimg):
        nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
        mu_g = np.average(nimg[1])
        nimg[0] = np.minimum(nimg[0] * (mu_g / np.average(nimg[0])), 255)
        nimg[2] = np.minimum(nimg[2] * (mu_g / np.average(nimg[2])), 255)
        return nimg.transpose(1, 2, 0).astype(np.uint8)

    def max_white(self,nimg):
        if nimg.dtype == np.uint8:
            brightest = float(2 ** 8)
        elif nimg.dtype == np.uint16:
            brightest = float(2 ** 16)
        elif nimg.dtype == np.uint32:
            brightest = float(2 ** 32)
        else:
            brightest = float(2 ** 8)
        nimg = nimg.transpose(2, 0, 1)
        nimg = nimg.astype(np.int32)
        nimg[0] = np.minimum(nimg[0] * (brightest / float(nimg[0].max())), 255)
        nimg[1] = np.minimum(nimg[1] * (brightest / float(nimg[1].max())), 255)
        nimg[2] = np.minimum(nimg[2] * (brightest / float(nimg[2].max())), 255)
        return nimg.transpose(1, 2, 0).astype(np.uint8)

    def retinex(self,nimg):
        nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
        mu_g = nimg[1].max()
        nimg[0] = np.minimum(nimg[0] * (mu_g / float(nimg[0].max())), 255)
        nimg[2] = np.minimum(nimg[2] * (mu_g / float(nimg[2].max())), 255)
        return nimg.transpose(1, 2, 0).astype(np.uint8)

    def retinex_adjust(self,nimg):
        """
        from 'Combining Gray World and Retinex Theory for Automatic White Balance in Digital Photography'
        """
        nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
        sum_r = np.sum(nimg[0])
        sum_r2 = np.sum(nimg[0] ** 2)
        max_r = nimg[0].max()
        max_r2 = max_r ** 2
        sum_g = np.sum(nimg[1])
        max_g = nimg[1].max()
        coefficient = np.linalg.solve(np.array([[sum_r2, sum_r], [max_r2, max_r]]),
                                      np.array([sum_g, max_g]))
        nimg[0] = np.minimum((nimg[0] ** 2) * coefficient[0] + nimg[0] * coefficient[1], 255)
        sum_b = np.sum(nimg[1])
        sum_b2 = np.sum(nimg[1] ** 2)
        max_b = nimg[1].max()
        max_b2 = max_r ** 2
        coefficient = np.linalg.solve(np.array([[sum_b2, sum_b], [max_b2, max_b]]),
                                      np.array([sum_g, max_g]))
        nimg[1] = np.minimum((nimg[1] ** 2) * coefficient[0] + nimg[1] * coefficient[1], 255)
        return nimg.transpose(1, 2, 0).astype(np.uint8)

    def retinex_with_adjust(self,nimg):
        return self.retinex_adjust(self.retinex(nimg))

    def gimp(self, img, perc=0.05):
        for channel in range(img.shape[2]):
            mi, ma = (np.percentile(img[:, :, channel], perc), np.percentile(img[:, :, channel], 100.0 - perc))
            img[:, :, channel] = np.uint8(np.clip((img[:, :, channel] - mi) * 255.0 / (ma - mi), 0, 255))
        return img

def main():
    try:
        global images_in

        images_in = images_in + '/data5_.data'

        imrows = 2464
        imcols = 3296

        imsize = imrows*imcols

        with open(images_in, "rb") as rawimage:
            img = np.fromfile(rawimage, np.dtype('u2'), imsize).reshape((imrows, imcols))

        db_rgb = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)

        wb = WhiteBalance()

        out = wb.gray_world(db_rgb)
        out_s = cv2.resize(out, (1098, 820))
        cv2.imshow("gray_world", out_s)

        '''
        out = wb.retinex_adjust(db_rgb)
        out_s = cv2.resize(out, (1098, 820))
        cv2.imshow("retinex_adjust", out_s)
      
        #db_rgb = (db_rgb - np.min(db_rgb)) / (np.max(db_rgb) - np.min(db_rgb))  # normalize image
        out = wb.gimp(db_rgb, 0.05) #0.05
        out_s = cv2.resize(out, (1098, 820))
        cv2.imshow("gimp", out_s)
          '''

        db_rgb = (db_rgb - np.min(db_rgb)) / (np.max(db_rgb) - np.min(db_rgb))  # normalize image
        db_rgb_s = cv2.resize(db_rgb,(1098,820))
        cv2.imshow("Orginal", db_rgb_s)

        cv2.waitKey(0)

    except Exception as e:
        print('Error in Main: ' + str(e))


if __name__ == '__main__':
    main()