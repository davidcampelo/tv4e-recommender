# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.utils import timezone
from django.db import models

from view.models import Boxes, InformativeVideos


class Senior(models.Model):
    name = models.CharField(max_length=255, null=False)
    email = models.CharField(max_length=255, null=False)
    gender = models.CharField(max_length=1, null=False) # M / F
    household_count = models.IntegerField(null=False)
    marital_status = models.CharField(max_length=15, null=True)
    box = models.ForeignKey(Boxes) 
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)


    def __str__(self):
        return "name: {}, email: {}, gender: {}, household_count: {}, marital_status: {}"\
            .format(self.name, self.email, self.gender, self.household_count, self.marital_status)

class Rating(models.Model):
    user_id = models.IntegerField(null=False)
    content_id = models.IntegerField(null=False)
    rating = models.DecimalField(decimal_places=1, max_digits=2)
    rating_timestamp = models.DateTimeField(default=timezone.now())
    rating_type = models.CharField(max_length=8, default='explicit')

    @property
    def content(self):
        return InformativeVideos.objects.get(pk=self.content_id)

    def __str__(self):
        return "user_id: {}, content_id: {}, rating: {}, type: {} rating_timestamp: {}"\
            .format(self.user_id, self.content_id, self.rating, self.rating_type, self.rating_timestamp)


