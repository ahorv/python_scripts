#from django.conf.urls import patterns, url
from django.conf.urls import *   # hoa


from djpilapp import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url('^shoot/(\d+)/(\d+)/$', views.shoot, name='shoot'),
    url('^findinitialparams/$', views.findinitialparams, name='findinitialparams'),
    url('^jsonupdate/$', views.jsonupdate, name='jsonupdate'),
    url('^newProject/$', views.newProjectSubmit, name='newProjectSubmit'),
    ## add URL for newProject view
    url('^saveproj/$', views.saveProjectSettings, name='saveProjectSettings'),
    url('^startlapse/$', views.startlapse, name='startlapse'),
    url('^deactivate/$', views.deactivate, name='deactivate'),
    url('^reboot/$', views.reboot, name='reboot'),
    url('^poweroff/$', views.poweroff, name='poweroff'),
    url('^deleteall/$', views.deleteall, name='deleteall'),
    url('^newProjectSubmit/$', views.newProjectSubmit, name='newProjectSubmit'),
]
