# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.db import models


class Asgie(models.Model):
    id = models.IntegerField(primary_key=True)
    title = models.CharField(unique=True, max_length=250)
    title_pt = models.CharField(max_length=100)
    image = models.CharField(max_length=200)
    color = models.CharField(max_length=100)

    def __str__(self):
        return '[{}] {}'.format(self.id, self.title)


class User(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=255)
    age = models.IntegerField(null=False)
    gender = models.CharField(max_length=1, null=False) # M / F
    city_id = models.IntegerField(null=False)
    coordinates = models.CharField(max_length=255, null=True)

    def __str__(self):
        return "id: {}, name: {}, age: {}, gender: {}, city_id: {}, coordinates: {}"\
            .format(self.id, self.name, self.age, self.gender, self.city_id, self.coordinates)


class Video(models.Model):
    id = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=500)
    # ALTER TABLE `tv4e`.`majordomo_video` CHANGE COLUMN `desc`
    # `desc` LONGTEXT CHARACTER SET 'utf8' NOT NULL ;
    desc = models.TextField()
    date_creation = models.DateTimeField(blank=False, null=False)
    location = models.CharField(max_length=1, null=True, blank=True)
    asgie = models.ForeignKey(Asgie, null=False)
    tokens = models.TextField(null=False)

    def __str__(self):
        return "id: {}, title: {}, "\
            .format(self.id, self.title)


class Rating(models.Model):
    user = models.ForeignKey(User, null=False)
    video = models.ForeignKey(Video, null=False)
    # Percentage of video watched
    watch_time = models.IntegerField(null=False)
    # Value of explicit rating
    value = models.DecimalField(decimal_places=1, max_digits=2, default=0)
    # Date when the user rated or watched
    date_creation = models.DateTimeField(null=False)
    # If the video was forced to be watched or notified
    watched_type = models.CharField(max_length=15)
    rating_implicit = models.DecimalField(decimal_places=9, max_digits=10, default=0)
    rating_explicit = models.DecimalField(decimal_places=9, max_digits=10, default=0)
    overall_rating_value = models.DecimalField(decimal_places=9, max_digits=10, default=0)

    def __str__(self):
        return "user_id: {}, video_id: {}, watch_time: {}, value: {}, date_creation: {}, watched_tyoe: {}"\
            .format(self.user.id, self.video.id, self.watch_time, self.value, self.date_creation, self.watched_type)
