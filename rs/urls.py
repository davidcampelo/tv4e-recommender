from django.conf.urls import url

from . import views

urlpatterns = [
    # ex: /rs/chart/
    #url(r'^chart/', views.chart, name='chart'),
    # ex: /rs/cb/content/98860/
    url(r'^cb/content/(?P<content_id>\w+)/$', views.similar_content, name='similar_content'),
]
