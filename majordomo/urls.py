from django.conf.urls import url

from . import views

urlpatterns = [
    # ex: /majordomo/refresh_recommendations/
    url(r'^refresh_recommendations', views.refresh_recommendations, name='refresh_recommendations'),
    # ex: /majordomo/similar_content/778/
    url(r'^similar_content/(?P<content_id>\w+)/$', views.similar_content, name='similar_content'),
    # ex: /majordomo/user_recommendations/1/
    url(r'^user_recommendations/(?P<user_id>\w+)/$', views.user_recommendations, name='user_recommendations'),
    # ex: /majordomo/fast_user_recommendations/1/
    url(r'^fast_user_recommendations/(?P<user_id>\w+)/$', views.fast_user_recommendations, name='fast_user_recommendations'),
    # analytics data
    url(r'^analytics/get_statistics', views.get_statistics, name='get_statistics'),
    url(r'^analytics/ratings_distribution', views.ratings_distribution, name='ratings_distribution'),
    url(r'^analytics/top10', views.top10, name='top10'),
    # graphs
    url(r'^img/user_ratings', views.img_user_ratings, name='img_user_ratings'),
    url(r'^img/rating_types', views.img_rating_types, name='img_rating_types'),
    url(r'^img/rating_dailyevolution', views.img_rating_dailyevolution, name='img_rating_dailyevolution'),  
    url(r'^img/rating_weekday', views.img_rating_weekday, name='img_rating_weekday'),
    url(r'^img/rating_hour', views.img_rating_hour, name='img_rating_hour'),
    url(r'^img/rating_evolution', views.img_rating_evolution, name='img_rating_evolution'),
    url(r'^img/rating_correlation', views.img_rating_correlation, name='img_rating_correlation'),
    url(r'^img/user_rating_correlation/(?P<user_id>\w+)/$', views.img_user_rating_correlation, name='img_user_rating_correlation'),
]
