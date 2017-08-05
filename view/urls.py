from django.conf.urls import url	

from . import views

urlpatterns = [
    # ex: /video/
    url(r'^video/$', views.index, name='video_index'),
	# ex: /videos/asgie/
	url(r'^asgie/$', views.asgie_index, name='asgie_index'),
	# ex: /videos/asgie/5/
	url(r'^asgie/(?P<asgie_id>[0-9]+)/$', views.asgie_detail, name='asgie_detail'),
]