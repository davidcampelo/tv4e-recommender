from django.conf.urls import url	

from . import views

urlpatterns = [
    # ex: /
    url(r'^$', views.index, name='view_index'),
    # ex: /video/
    url(r'^video/$', views.VideoIndexView.as_view(), name='video_index'),
    # ex: /video/
    url(r'^video/(?P<video_id>[0-9]+)/$', views.video_detail, name='video_detail'),
	# ex: /videos/informativevideo/
	url(r'^asgie/$', views.AsgieIndexView.as_view(), name='asgie_index'),
	# ex: /videos/informativevideo/5/
	url(r'^asgie/(?P<asgie_id>[0-9]+)/$', views.asgie_detail, name='asgie_detail'),
]