from django.conf.urls import patterns, include, url

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'djpilapse.views.home', name='home'),
    # url(r'^djpilapse/', include('djpilapse.foo.urls')),

    url(r'^djpilapp/', include('djpilapp.urls')),

    url(r'^$', include('djpilapp.urls')),
    url(r'^/$', include('djpilapp.urls')),


    # Uncomment the admin/doc line below to enable admin documentation:
    url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    url(r'^admin/', include(admin.site.urls)),
)
