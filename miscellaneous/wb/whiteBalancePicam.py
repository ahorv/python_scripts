import picamera
import picamera.array
import numpy as np

'''
Source: https://raspberrypi.stackexchange.com/questions/22975/custom-white-balancing-with-picamera

You could write a little loop that assumes that the camera is pointed at something 
which is mostly white (e.g. a sheet of paper) and then iterate over various combinations 
of the red and blue gains (probably increments of 0.1 between, say, 0.5 and 2.5) 
until you find the combination that produces an image in which most of the pixels are as 
close to grey (i.e. equal values for R, G, and B) as possible.

In my tests it usually gets close to a decent solution in 10 or so steps, and then wobbles 
around a couple of values. There's almost certainly betters ways of doing this (starting with more 
sensible values, varying one at a time, using YUV captures instead, decreasing the increments as the 
values converge, terminating when acceptably close, etc.) but this should be enough to demonstrate the principle.

'''

with picamera.PiCamera() as camera:
    camera.resolution = (1280, 720)
    camera.awb_mode = 'off'
    # Start off with ridiculously low gains
    rg, bg = (0.5, 0.5)
    camera.awb_gains = (rg, bg)
    with picamera.array.PiRGBArray(camera, size=(128, 72)) as output:
        # Allow 30 attempts to fix AWB
        for i in range(30):
            # Capture a tiny resized image in RGB format, and extract the
            # average R, G, and B values
            camera.capture(output, format='rgb', resize=(128, 72), use_video_port=True)
            r, g, b = (np.mean(output.array[..., i]) for i in range(3))
            print('R:%5.2f, B:%5.2f = (%5.2f, %5.2f, %5.2f)' % (rg, bg, r, g, b))
            # Adjust R and B relative to G, but only if they're significantly
            # different (delta +/- 2)
            if abs(r - g) > 2:
                if r > g:
                    rg -= 0.1
                else:
                    rg += 0.1
            if abs(b - g) > 1:
                if b > g:
                    bg -= 0.1
                else:
                    bg += 0.1
            camera.awb_gains = (rg, bg)
            output.seek(0)
            output.truncate()