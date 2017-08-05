from django.conf.urls import url	

from . import views

urlpatterns = [
    # ex: /videos/
    url(r'^$', views.index, name='index'),
    # ex: /videos/5/
    url(r'^(?P<video_id>[0-9]+)/$', views.video_detail, name='video_detail'),
	# ex: /videos/genre/5/
	url(r'^asgie/(?P<asgie_id>[0-9]+)/$', views.asgie_detail, name='asgie_detail'),
]