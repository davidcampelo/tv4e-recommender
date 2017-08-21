# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from view.models import InformativeVideos

import redis

def similar_content(request, content_id):
    # XXX Put url in a config file! 
    db = redis.StrictRedis.from_url('redis://localhost:6379')
    similarities = db.lrange("content_similarity:%s" % content_id, 0, 3)
    print(similarities)
    columns = ['target_id', 'target_img', 'target_title', 'confidence']
    data = []
    for similar in similarities:
    	similar = similar.split(' ')
    	video_id = similar[0]
        video = InformativeVideos.objects.get(pk=video_id)
    	confidence = round(float(similar[1]), 2)
    	title = InformativeVideos.objects.values('title').filter(id=video_id)[0]['title']
    	data.append({columns[0]:video_id, columns[1]:video.asgie.image, columns[2]:title, columns[3]:confidence})

    return JsonResponse(dict(data=list(data)), safe=False)


