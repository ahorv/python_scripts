from __future__ import absolute_import
from celery import shared_task
from djpilapp.models import timelapser
import os, subprocess
from time import time, sleep
from PIL import Image

@shared_task
def add(x, y):
    return x + y

@shared_task
def timelapse(width=20):
    #try:
    T=timelapser.objects.all()[0]
    T.set_status('Timelapse active')
    L=[T.project.brightness]
    while T.active:
        loopstart=time()
        T=timelapser.objects.all()[0]
        proj=T.project
        L=timelapse_shoot(L, width)
        if not T.active: break
        loopend=time()
        sleep(max([0,proj.interval-(loopend-loopstart)]))
    #except:
    #    T.set_active(False)
    #    T.set_status('idle')
    #    return False
    T.set_status('idle')
    T.set_active(False)
    return True

@shared_task
def timelapse_shoot(L=None, width=20, gamma=None):
    """
    `L` is a list of recent image brightnesses.
    `width` is the number of images to use in finding average brightness.
    """
    T=timelapser.objects.all()[0]
    if not T.active: return None
    proj=T.project
    if L==None: L=[proj.brightness]
    if gamma==None: gamma=1.0/width

    #figure out the filename.
    dtime=subprocess.check_output(['date', '+%y%m%d_%T']).strip()
    dtime=dtime.replace(':', '.')
    filename=proj.folder
    if filename[-1]!='/': filename+='/'
    filename+= proj.project_name + '_' + dtime + '.jpg'
    tempfile='/home/pi/python_scripts/pipic/djpilapse/djpilapp/static/new.jpg'

    #Take a picture
    options='-awb auto -n'
    options+=' -w '+str(proj.width)+' -h '+str(proj.height)
    options+=' -t 50'
    options+=' -ss '+str(T.ss)
    options+=' -ISO '+str(T.iso)
    options+=' -o '+tempfile
    try:
        subprocess.call('raspistill '+options, shell=True)
        print('raspistill {}'.format(options))
        im=Image.open(tempfile)
        #Saves file without exif and raster data; reduces file size by 90%,
        if filename!=None:
            im.save(filename)
        print('{}'.format(filename))
    except:
        return False

    newbr=T.avgbrightness(im)
    if len(L)>=width: L=L[1:]
    L.append(newbr)
    avgbr=sum(L)/len(L)
    T.lastbr=newbr
    T.avgbr=avgbr

    #Dynamically adjust ss and iso.
    (T.ss, T.iso)=T.dynamic_adjust(target=proj.brightness,
                                   lastbr=avgbr, gamma=gamma)
    print('{}'.format(str(L)))
    print('{} \t{} \t{} \t{}'.format(str(newbr),str(avgbr),str(T.ss),str(T.iso)))
    T.shots_taken+=1

    delta=proj.brightness-avgbr
    #if abs(delta)>self.maxdelta and not (maxxedbr or minnedbr):
    if abs(delta)>proj.delta:
        #Too far from target brightness.
        T.shots_taken-=1
        os.remove(filename)
    else:
        T.lastshot=filename
    T1=timelapser.objects.all()[0]
    if not T1.active: return None
    T.save()
    return L

