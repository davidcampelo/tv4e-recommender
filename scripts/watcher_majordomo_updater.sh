#!/bin/bash

while true
do
    /home/ubuntu/tv4e-recommender/scripts/majordomo_updater_run.sh >> /home/ubuntu/tv4e-recommender/logs/watcher_majordomo_logs.log 2>&1
    sleep 900 # 15 minutes
done
