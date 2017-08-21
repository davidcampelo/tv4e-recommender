from django.conf.urls import url	

from . import views

urlpatterns = [
    # ex: view/
    url(r'^$', views.video_index, name='view_index'),
    # ex: view/video/
    url(r'^video/$', views.video_index, name='video_index'),
    # ex: view/video/333/
    url(r'^video/(?P<video_id>[0-9]+)/$', views.video_detail, name='video_detail'),
    # ex: /videos/asgie/
    url(r'^asgie/$', views.AsgieIndexView.as_view(), name='asgie_index'),
    # ex: /videos/asgie/5/
    url(r'^asgie/(?P<asgie_id>[0-9]+)/$', views.asgie_detail, name='asgie_detail'),
    # ex: /videos/user/
    url(r'^user/$', views.UserIndexView.as_view(), name='user_index'),
    # ex: /videos/user/5/
    url(r'^user/(?P<user_id>[0-9]+)/$', views.user_detail, name='user_detail'),
]