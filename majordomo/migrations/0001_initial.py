# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2017-11-28 20:48
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Asgie',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('title', models.CharField(max_length=250, unique=True)),
                ('title_pt', models.CharField(max_length=100)),
                ('image', models.CharField(max_length=200)),
                ('color', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='City',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=200)),
            ],
        ),
        migrations.CreateModel(
            name='Rating',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('watch_time', models.IntegerField()),
                ('date_creation', models.DateTimeField()),
                ('watched_type', models.CharField(max_length=15)),
                ('rating_implicit', models.DecimalField(decimal_places=9, default=0, max_digits=10)),
                ('rating_explicit', models.DecimalField(decimal_places=9, default=None, max_digits=10, null=True)),
                ('overall_rating_value', models.DecimalField(decimal_places=9, default=0, max_digits=10)),
            ],
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=255)),
                ('age', models.IntegerField()),
                ('gender', models.CharField(max_length=1)),
                ('coordinates', models.CharField(max_length=255, null=True)),
                ('city', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='majordomo.City')),
            ],
        ),
        migrations.CreateModel(
            name='Video',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('title', models.CharField(max_length=500)),
                ('desc', models.TextField()),
                ('date_creation', models.DateTimeField()),
                ('location', models.CharField(blank=True, max_length=1, null=True)),
                ('tokens', models.TextField()),
                ('asgie', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='majordomo.Asgie')),
            ],
        ),
        migrations.AddField(
            model_name='rating',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='majordomo.User'),
        ),
        migrations.AddField(
            model_name='rating',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='majordomo.Video'),
        ),
    ]
