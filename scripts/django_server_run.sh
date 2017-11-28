#!/bin/bash

cd /home/ubuntu/tv4e-recommender/ 
source tv4e_project/bin/activate 
python manage.py runserver 79.137.39.168:8080
