# encoding: utf-8
import logging
import random
import django
import time
import decimal

# XXX work around to run this script in a sub-directory of the project (/scripts)
import os
import sys
root_path = os.path.abspath(os.path.split(__file__)[0])
sys.path.insert(0, os.path.join(root_path, '../'))
sys.path.insert(0, root_path)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tv4e.settings")
django.setup()

import numpy as np
import pandas as pd
from view.models import Asgie, InformativeVideos, AsgieAvResource, Boxes
from rs.models import Senior, Rating


RAW_DATA_FILENAME = 'ratings.csv'
ratings = Rating.objects.all()
data = np.array([[rating.user_id,  rating.content_id, rating.rating, rating.rating_timestamp, rating.rating_type] for rating in ratings])
dataframe = pd.DataFrame(data=data[0:,0:], index=data[0:,0], columns=['user_id', 'content_id', 'rating', 'rating_timestamp', 'rating_type'])
print("Saving raw data to CSV [%s]..." % RAW_DATA_FILENAME)
dataframe.to_csv(RAW_DATA_FILENAME, encoding='utf-8', sep=',', index=False)
