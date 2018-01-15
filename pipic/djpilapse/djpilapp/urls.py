from django.conf.urls import url

from djpilapp import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^shoot/(\d+)/(\d+)/$', views.shoot, name='shoot'),
    url(r'^findinitialparams/$', views.findinitialparams, name='findinitialparams'),
    url(r'^jsonupdate/$', views.jsonupdate, name='jsonupdate'),
    url(r'^newProject/$', views.newProjectSubmit, name='newProjectSubmit'),
    ## add URL for newProject view
    url(r'^saveproj/$', views.saveProjectSettings, name='saveProjectSettings'),
    url(r'^startlapse/$', views.startlapse, name='startlapse'),
    url(r'^deactivate/$', views.deactivate, name='deactivate'),
    url(r'^reboot/$', views.reboot, name='reboot'),
    url(r'^poweroff/$', views.poweroff, name='poweroff'),
    url(r'^deleteall/$', views.deleteall, name='deleteall'),
    url(r'^newProjectSubmit/$', views.newProjectSubmit, name='newProjectSubmit'),
]
