from django.conf.urls import url

from . import views

urlpatterns = [
    # ex: /majordomo/refresh_recommendations/
    url(r'^refresh_recommendations', views.refresh_recommendations, name='refresh_recommendations'),
    # ex: /majordomo/similar_content/778/
    url(r'^similar_content/(?P<content_id>\w+)/$', views.similar_content, name='similar_content'),
    # ex: /majordomo/user_recommendations/1/
    url(r'^user_recommendations/(?P<user_id>\w+)/$', views.user_recommendations, name='user_recommendations'),
    # ex: /majordomo/user_recommendations/1/
    url(r'^fast_user_recommendations/(?P<user_id>\w+)/$', views.fast_user_recommendations, name='fast_user_recommendations'),
    # # ex: /majordomo/analytics/get_statistics
    # url(r'^analytics/get_statistics', views.get_statistics, name='get_statistics'),
    # # ex: /majordomo/analytics/ratings_distribution
    # url(r'^analytics/ratings_distribution', views.ratings_distribution, name='ratings_distribution'),
    # # ex: /majordomo/analytics/ratings_dailyevolution
    # url(r'^analytics/ratings_dailyevolution', views.ratings_dailyevolution, name='ratings_dailyevolution'),
    # # ex: /majordomo/analytics/ratings_weekday
    # url(r'^analytics/ratings_weekday', views.ratings_weekday, name='ratings_weekday'),
    # # ex: /majordomo/analytics/top10
    # url(r'^analytics/top10', views.top10, name='top10'),

]
