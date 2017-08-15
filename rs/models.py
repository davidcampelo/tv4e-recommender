# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

class Rating(models.Model):
    user_id = models.IntegerField()
    content_id = models.IntegerField()
    rating = models.DecimalField(decimal_places=2, max_digits=4)
    rating_timestamp = models.DateTimeField()
    type = models.CharField(max_length=8, default='explicit')

    def __str__(self):
        return "user_id: {}, movie_id: {}, rating: {}, type: {}"\
            .format(self.user_id, self.movie_id, self.rating, self.type)
