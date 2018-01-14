# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseServerError, JsonResponse
from django.conf import settings
from django.db.models import Avg, Count
from django.db import connection


import operator
import redis
import logging
import traceback

from majordomo.models import Video,Rating,User
from majordomo.updater import Updater
from majordomo.lock import AlreadyLockedError

logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s', level=logging.DEBUG)


def refresh_recommendations(request):
    updater = Updater()
    error = False
    try:
        updater.lock()
        updater.update_recommendations()
    except AlreadyLockedError as err:
        error = True
        logging.warning("***** Called refresh during refresh recommendations!")
        return HttpResponseServerError(err)
    except Exception as err:
        error = True
        logging.error("***** Unhandled error during refresh_recommendations: {}". format(type(err)))
        traceback.print_exc()
        return HttpResponseServerError(err)

    if not error:
            updater.unlock()
            return HttpResponse("Ok")

def similar_content(request, content_id):
    db = redis.StrictRedis.from_url(settings.REDIS_URL)
    key = "%s%s%s" % (settings.KEY_CONTENT_SIMILARITY, settings.SEPARATOR, content_id)
    similarities = db.lrange(key, 0, settings.NUMBER_OF_RECOMMENDATIONS - 1)

    columns = ['video_id', 'target_title', 'asgie_image', 'confidence']
    data = []
    for similar in similarities:
        similar = similar.decode('utf-8').split(settings.SEPARATOR)
        video_id = similar[0]
        video = Video.objects.get(pk=video_id)
        confidence = round(float(similar[1]), 2)
        data.append({
            columns[0]: video.id,
            columns[1]: video.title,
            columns[2]: video.asgie.image,
            columns[3]: confidence
        })

    return JsonResponse(dict(data=list(data)), safe=False)

def user_recommendations(request, user_id): 
    db = redis.StrictRedis.from_url(settings.REDIS_URL)
    key = "%s%s%s" % (settings.KEY_USER_RECOMMENDATION, settings.SEPARATOR, user_id)
    user_recommendations = db.lrange(key, 0, settings.NUMBER_OF_RECOMMENDATIONS - 1)
    columns = ['video_id', 'video_title', 'asgie_image']
    data = []
    for video_id in user_recommendations:
        video = Video.objects.get(pk=video_id)
        data.append({
            columns[0]: video.id,
            columns[1]: video.title,
            columns[2]: video.asgie.image
        })
    logging.debug("***** Returning user_recommendations user_id={} recommendations={}".format(user_id, ' '.join([video_id.decode('utf-8') for video_id in user_recommendations])))
    return JsonResponse(dict(data=list(data)), safe=False)

def fast_user_recommendations(request, user_id):
    db = redis.StrictRedis.from_url(settings.REDIS_URL)
    key = "%s%s%s" % (settings.KEY_USER_RECOMMENDATION, settings.SEPARATOR, user_id)
    user_recommendations = db.lrange(key, 0, settings.NUMBER_OF_RECOMMENDATIONS - 1)
    columns = ['user_id', 'video_id']
    data = [{columns[0]: user_id, columns[1]: video_id.decode('utf-8')} for video_id in user_recommendations]
    logging.debug("***** Returning fast_user_recommendations user_id={} recommendations={}".format(user_id, ' '.join([video_id.decode('utf-8') for video_id in user_recommendations])))

    return JsonResponse(dict(data=list(data)), safe=False)

def get_statistics(request):
    # XXX filter statistics from last month
    # date_timestamp = time.strptime(request.GET["date"], "%Y-%m-%d")
    # end_date = datetime.fromtimestamp(time.mktime(date_timestamp))
    # start_date = monthdelta(end_date, -1)
    # print("getting statics for ", start_date, " and ", end_date)

    # number of videos rated
    videos_rated_distinct = Rating.objects.values('video_id').distinct().count()
    videos_rated_total = Rating.objects.values('video_id').count()

    # number of active users
    n_active_users = Rating.objects.values('user_id').distinct().count()
    n_total_users = User.objects.values('id').distinct().count()
    active_users_percent = round(n_active_users*100/float(n_total_users), 2)

    # mean overall rating
    mean_rating = round(Rating.objects.aggregate(avg=Avg('overall_rating_value'))['avg'], 3)

    # most active user (max number of ratings)
    # n_max_ratings = Rating.objects.values('user_id').annotate(count=Count('user_id')).aggregate(max=Max('count'))[0]
    ratings_users = Rating.objects.values('user_id').annotate(count=Count('user_id'))
    user = sorted(ratings_users, key=operator.itemgetter('count'))[:-2:-1][0]
    most_active_user_count = user['count']
    most_active_user_id = user['user_id']
    most_active_user_name = User.objects.filter(pk=most_active_user_id).values('name')[0]['name']

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
         "most_active_user_id": most_active_user_id,
         "most_active_user_name": most_active_user_name,
         "most_active_user_count": most_active_user_count,
         "n_active_users" : n_active_users,
         "n_total_users" : n_total_users,
         "active_users_percent": active_users_percent})


def dictfetchall(cursor):
    " Returns all rows from a cursor as a dict "
    desc = cursor.description
    return [
        dict(zip([col[0] for col in desc], row))
        for row in cursor.fetchall()
        ]

#
def ratings_distribution(request):
    cursor = connection.cursor()
    cursor.execute("""
    select round(overall_rating_value,1) as classificação, count(*) as quantidade
    from majordomo_rating
    group by classificação
    order by classificação
    """)
    data = dictfetchall(cursor)
    print(data)
    return JsonResponse(data, safe=False)
#
#
def ratings_dailyevolution(request):
    cursor = connection.cursor()
    cursor.execute("""
    select day(date_creation) as dia, count(*) as quantidade
    from majordomo_rating
    group by day(date_creation)
    """)
    data = dictfetchall(cursor)

    print("data = {}".format(data))
    data2 = []
    dias_do_mes = [i for i in range(1,31+1)]
    for index,dia in enumerate(dias_do_mes):
        data2.append(({'dia': index+1, 'quantidade': 0}))
    # print("data2 = {}".format(data2))

    for index, item in enumerate(data):
        data2[item['dia']-1]['quantidade'] = item['quantidade']
    # print("data2 = {}".format(data2))


    data3 = []
    acumulado = 0
    for index,item in enumerate(data2):
        acumulado += item['quantidade']
        # print("dia = %s count = %s acumulado = %s" % (item['dia'], item['quantidade'], acumulado))
        data3.append({'dia': item['dia'], 'quantidade': item['quantidade'], 'acumulado': acumulado})
    # print("data3 = {}".format(data2))

    return JsonResponse(data3, safe=False)


#
def ratings_weekday(request):
    cursor = connection.cursor()
    cursor.execute("""
    select weekday(date_creation) as dia_da_semana_num, count(*) as quantidade
    from majordomo_rating
    group by weekday(date_creation)
    """)
    data = dictfetchall(cursor)

    # XXX Work-around to include weekdays where no ratings where found
    data2 = []
    dias_da_semana = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    for index,dia in enumerate(dias_da_semana):
        data2.append(({'dia_da_semana_num': index, 'quantidade': 0, 'dia da semana': dia[:3]}))

    for index, item in enumerate(data):
        data2[item['dia_da_semana_num']]['quantidade'] = item['quantidade']

    return JsonResponse(data2, safe=False)

def top10(request):
    top10 = Rating.objects.values('video_id').annotate(avg=Avg('overall_rating_value')).order_by('-avg')[:10]
    columns = ['video_id', 'video_title', 'avg_rating']
    data = []

    for top in top10:
        video_id = top['video_id']
        video = Video.objects.get(pk=video_id)
        title = Video.objects.values('title').filter(id=video_id)[0]['title']
        data.append({columns[0]:video_id, columns[1]:video.title, columns[2]:top['avg']})

    return JsonResponse(dict(data=list(data)), safe=False)

