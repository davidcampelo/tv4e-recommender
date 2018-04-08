# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseServerError, JsonResponse
from django.conf import settings
from django.db.models import Avg, Count, Q
from django.db import connection


import operator
import redis
import logging
import traceback

from majordomo.models import Video,Rating,User
from majordomo.updater import Updater
from majordomo.lock import AlreadyLockedError

logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s', level=logging.DEBUG)

LIST_OF_DEV_USER_ID = ['1', '2', '3', '9', '21']

def img_rating_dailyevolution(request):
    import matplotlib
    import matplotlib.dates as mdates
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib import style
    from collections import Counter
    import datetime as dt
    style.use('ggplot')

    # Grouping ratings by user
    ratings_query = Rating.objects.exclude(Q(watched_type='forced') | Q(user_id__in=LIST_OF_DEV_USER_ID))
    df_ratings = pd.DataFrame(list(ratings_query.values()))
    df_ratings['date_creation2'] = pd.to_datetime(df_ratings['date_creation']).dt.strftime('%m/%d/%Y')
    d = Counter(df_ratings['date_creation2'])
    df = pd.DataFrame(list(d.items()), columns=['date_creation2', 'Count'])
    df = df.set_index('date_creation2').sort_index()
    df.index = pd.to_datetime(df.index)

    # Plot figure
    fig = Figure()
    ax1=fig.add_subplot(1,1,1)
    ax1.set_title('Evolução diária', fontsize='medium')
    ax2 = ax1.twinx()

    ax1.bar(df.index, df.Count, color='#428bca')
    ax2.plot(df.index, df.Count.cumsum(), color='#800000', label='Acumulado')
    ax2.plot([], [], color='#428bca', label='Diária')
    ax2.legend(loc='best', fontsize='small')
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(90)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
    ax1.xaxis.set_minor_formatter(mdates.DateFormatter("%d/%m/%Y"))

    fig.subplots_adjust(bottom=0.22, left=0.08)
    # fig.tight_layout()
    ax1.set_ylabel('Quantidade diária', fontsize='small')
    ax2.set_ylabel('Quantidade acumulada', fontsize='small')
    ax1.set_xlabel('Data', fontsize='small')
    ax1.grid(True)

    # Create response object
    canvas = FigureCanvas(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)

    return response


def img_rating_types(request):
    import matplotlib
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib import style
    style.use('ggplot')

    # Grouping ratings by user
    ratings_query = Rating.objects.exclude(Q(watched_type='forced') | Q(user_id__in=LIST_OF_DEV_USER_ID))
    df_ratings = pd.DataFrame(list(ratings_query.values()))
    df_ratings.set_index('id', inplace=True)
    df_ratings_distribution = df_ratings.fillna(-2) # using -2 to represent the scenario where no rating screen was shown
    df_ratings_distribution = df_ratings_distribution.groupby(['rating_explicit']).size().reset_index(name='counts')

    # Plot figure
    fig = Figure()
    ax=fig.add_subplot(1,1,1)
    ax.set_title('Uso do ecrã de classificação', fontsize='medium')
    ax.bar(range(len(df_ratings_distribution.rating_explicit)), df_ratings_distribution.counts, color='#428bca', width=0.8)
    ax.set_ylabel('Quantidade', fontsize='small')
    labels = ['Ecrã de classificação\n não exibido',
              'Ecrã exibido e \nvoto negativo recebido',
              'Ecrã exibido e \nvoto não recebido',
              'Ecrã exibido e \nvoto positivo recebido']

    ax.set_xticks(range(len(df_ratings_distribution.rating_explicit)))
    ax.set_xticklabels(labels, fontsize='small')
    ax.grid(True)

    # Create response object
    canvas = FigureCanvas(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)

    return response

def img_user_ratings(request):
    import matplotlib
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib import style
    style.use('ggplot')

    # Grouping ratings by user
    ratings_query = Rating.objects.exclude(Q(watched_type='forced') | Q(user_id__in=LIST_OF_DEV_USER_ID))
    df_ratings = pd.DataFrame(list(ratings_query.values()))
    df_ratings.set_index('id', inplace=True)
    df_ratings_by_user = df_ratings.groupby(['user_id']).size().reset_index(name='counts')
    id = df_ratings_by_user.user_id.unique().astype(str)

    # Plot figure
    fig = Figure()
    ax=fig.add_subplot(1,1,1)
    ax.set_title('Classificações por utilizador', fontsize='medium')
    ax.bar(range(len(id)), df_ratings_by_user.counts, color='#428bca', width=0.8)
    ax.set_ylabel('Quantidade', fontsize='small')
    ax.set_xlabel('ID', fontsize='small')
    ax.set_xticks(range(len(id)))
    ax.set_xticklabels(id)
    ax.grid(True)

    # Create response object
    canvas = FigureCanvas(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)

    return response


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
    ratings_query = Rating.objects.exclude(Q(watched_type='forced') | Q(user_id__in=LIST_OF_DEV_USER_ID))
    users_query = User.objects.exclude(Q(id__in=LIST_OF_DEV_USER_ID))
    # number of videos rated
    videos_rated_distinct = ratings_query.values('video_id').distinct().count()
    videos_rated_total = ratings_query.values('video_id').count()

    # number of active users
    n_active_users = ratings_query.values('user_id').distinct().count()
    n_total_users = users_query.values('id').distinct().count()
    active_users_percent = round(n_active_users*100/float(n_total_users), 2)

    # mean videos rated by user
    mean_videos_rated = round(videos_rated_total/n_active_users, 2)
    # mean overall rating
    mean_rating = round(ratings_query.aggregate(avg=Avg('overall_rating_value'))['avg'], 3)

    # precision
    negative_votes = ratings_query.filter(Q(rating_explicit=-1)).count()
    positive_votes = ratings_query.filter(Q(rating_explicit=1)).count()
    precision = round(positive_votes/(positive_votes + negative_votes) * 100, 2)

    # most active user (max number of ratings)
    # n_max_ratings = Rating.objects.values('user_id').annotate(count=Count('user_id')).aggregate(max=Max('count'))[0]
    ratings_users = ratings_query.values('user_id').annotate(count=Count('user_id'))
    user = sorted(ratings_users, key=operator.itemgetter('count'))[:-2:-1][0]
    most_active_user_count = user['count']
    most_active_user_id = user['user_id']
    most_active_user_name = users_query.filter(pk=most_active_user_id).values('name')[0]['name']

    return JsonResponse(
        {"videos_rated_distinct": videos_rated_distinct,
         "videos_rated_total": videos_rated_total,
         "mean_rating": mean_rating,
         "precision": precision,
         "mean_videos_rated": mean_videos_rated,
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
    WHERE NOT ((watched_type = 'forced' OR user_id IN ("""+ ','.join(LIST_OF_DEV_USER_ID) + """)))
    group by classificação
    order by classificação
    """)
    data = dictfetchall(cursor)
    print(data)
    return JsonResponse(data, safe=False)
#
#
def ratings_weekday(request):
    ratings_query = Rating.objects.exclude(Q(watched_type='forced') | Q(user_id__in=LIST_OF_DEV_USER_ID))
    weekday     = {"weekday": """weekday(date_creation)"""}
    ratings_weekday = ratings_query.extra(select=weekday).values('weekday').annotate(count=Count('id')).order_by('weekday')

    # XXX Work-around to include weekdays where no ratings where found
    data = []
    dias_da_semana = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    for index,dia in enumerate(dias_da_semana):
        data.append(({'dia_da_semana_num': index, 'quantidade': 0, 'dia da semana': dia[:3]}))

    for index, item in enumerate(ratings_weekday):
        data[item['weekday']]['quantidade'] = item['count']

    return JsonResponse(data, safe=False)

def top10(request):
    ratings_query = Rating.objects.exclude(Q(watched_type='forced') | Q(user_id__in=LIST_OF_DEV_USER_ID))
    top10 = ratings_query.values('video_id').annotate(avg=Avg('overall_rating_value')).order_by('-avg')[:10]
    columns = ['video_id', 'video_title', 'avg_rating']
    data = []

    for top in top10:
        video_id = top['video_id']
        video = Video.objects.get(pk=video_id)
        title = Video.objects.values('title').filter(id=video_id)[0]['title']
        data.append({columns[0]:video_id, columns[1]:video.title, columns[2]:top['avg']})

    return JsonResponse(dict(data=list(data)), safe=False)

