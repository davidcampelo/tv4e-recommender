# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2018-04-13 09:41
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('majordomo', '0003_auto_20180216_1527'),
    ]

    operations = [
        migrations.AddField(
            model_name='video',
            name='duration',
            field=models.IntegerField(null=True),
        ),
    ]
