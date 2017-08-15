# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse

def similar_content(request, content_id):
	return HttpResponse("OK %s" % content_id)

