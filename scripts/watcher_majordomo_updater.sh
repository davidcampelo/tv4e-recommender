#!/bin/bash

while true
do
    /home/ubuntu/majordomo_updater_run.sh >> /home/ubuntu/tv4e-recommender/watcher_majordomo_logs.log 2>&1 
    sleep 60
done
