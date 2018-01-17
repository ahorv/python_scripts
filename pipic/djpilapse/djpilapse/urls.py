from django.conf.urls import url, include
from django.urls import path
from django.contrib import admin


urlpatterns = [
    # Examples:
    # url(r'^$', 'djpilapse.views.home', name='home'),
    # url(r'^djpilapse/', include('djpilapse.foo.urls')),

    path('djpilapp/', include('djpilapp.urls')),


    # Uncomment the admin/doc line below to enable admin documentation:
    #url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    url(r'^admin/', admin.site.urls),
]
