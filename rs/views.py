# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from django.db.models import Avg, Count
from django.db import connection

from view.models import InformativeVideos
from rs.models import Rating, Senior

import operator
import redis

def similar_content(request, content_id):
    db = redis.StrictRedis.from_url(settings.REDIS_URL)
    key = "%s%s%s" % (settings.KEY_CONTENT_SIMILARITY, settings.SEPARATOR, content_id)
    similarities = db.lrange(key, 0, 3)

    columns = ['target_id', 'target_img', 'target_title', 'confidence']
    data = []
    for similar in similarities:
    	similar = similar.split(settings.SEPARATOR)
    	video_id = similar[0]
        video = InformativeVideos.objects.get(pk=video_id)
    	confidence = round(float(similar[1]), 2)
    	title = InformativeVideos.objects.values('title').filter(id=video_id)[0]['title']
    	data.append({columns[0]:video_id, columns[1]:video.asgie.image, columns[2]:title, columns[3]:confidence})

    return JsonResponse(dict(data=list(data)), safe=False)

def user_recommendations(request, user_id): 
    db = redis.StrictRedis.from_url(settings.REDIS_URL)
    key = "%s%s%s" % (settings.KEY_USER_RECOMMENDATION, settings.SEPARATOR, user_id)
    user_recommendations = db.lrange(key, 0, 3)
    columns = ['target_id', 'target_img', 'target_title']
    data = []
    for video_id in user_recommendations:
        video = InformativeVideos.objects.get(pk=video_id)
        title = InformativeVideos.objects.values('title').filter(id=video_id)[0]['title']
        data.append({columns[0]:video_id, columns[1]:video.asgie.image, columns[2]:title})

    return JsonResponse(dict(data=list(data)), safe=False)

def get_statistics(request):
    # XXX filter statistics from last month
    # date_timestamp = time.strptime(request.GET["date"], "%Y-%m-%d")
    # end_date = datetime.fromtimestamp(time.mktime(date_timestamp))
    # start_date = monthdelta(end_date, -1)
    # print("getting statics for ", start_date, " and ", end_date)

    # number of videos rated
    videos_rated_distinct = Rating.objects.values('content_id').distinct().count()
    videos_rated_total = Rating.objects.values('content_id').count()
    
    # number of active users
    n_active_users = Rating.objects.values('user_id').distinct().count()
    n_total_users = Senior.objects.values('id').distinct().count()
    active_users = round(n_active_users*100/float(n_total_users), 2)

    # mean overall rating 
    mean_rating = round(Rating.objects.aggregate(avg=Avg('rating'))['avg'], 1)

    # most active user (max number of ratings)
    # n_max_ratings = Rating.objects.values('user_id').annotate(count=Count('user_id')).aggregate(max=Max('count'))[0]
    ratings_users = Rating.objects.values('user_id').annotate(count=Count('user_id'))
    user_id = sorted(ratings_users, key=operator.itemgetter('count'))[:-2:-1][0]['user_id']
    most_active_user = Senior.objects.filter(pk=user_id).values('name')[0]['name']

    # sessions_with_conversions = Log.objects.filter(created__range=(start_date, end_date), event='buy') \
    #     .values('session_id').distinct()
    # buy_data = Log.objects.filter(created__range=(start_date, end_date), event='buy') \
    #     .values('event', 'user_id', 'content_id', 'session_id')
    # visitors = Log.objects.filter(created__range=(start_date, end_date)) \
    #     .values('user_id').distinct()
    # sessions = Log.objects.filter(created__range=(start_date, end_date)) \
    #     .values('session_id').distinct()

    # if len(sessions) == 0:
    #     conversions = 0
    # else:
    #     conversions = (len(sessions_with_conversions) / len(sessions)) * 100
    #     conversions = round(conversions)

    return JsonResponse(
        {"videos_rated_distinct": videos_rated_distinct,
         "videos_rated_total": videos_rated_total,
         "mean_rating": mean_rating,
         "most_active_user": most_active_user,
         "active_users": active_users})


def dictfetchall(cursor):
    " Returns all rows from a cursor as a dict "
    desc = cursor.description
    return [
        dict(zip([col[0] for col in desc], row))
        for row in cursor.fetchall()
        ]


def ratings_distribution(request):
    cursor = connection.cursor()
    cursor.execute("""
    select rating, count(1) as count_items
    from rs_rating
    group by rating
    order by rating
    """)
    data = dictfetchall(cursor)
    print(data)
    return JsonResponse(data, safe=False)


def top10(request):
    top10 = Rating.objects.values('content_id').annotate(avg=Avg('rating')).order_by('-avg')[:10]
    columns = ['video_id', 'video_title', 'avg_rating']
    data = []

    for top in top10:
        video_id = top['content_id']
        video = InformativeVideos.objects.get(pk=video_id)
        title = InformativeVideos.objects.values('title').filter(id=video_id)[0]['title']
        data.append({columns[0]:video_id, columns[1]:video.title, columns[2]:top['avg']})

    return JsonResponse(dict(data=list(data)), safe=False)

