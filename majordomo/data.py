# encoding: utf-8
import logging
import pandas as pd
import numpy as np
import requests
import redis
import dateutil
import pytz
import traceback
import datetime

from majordomo.models import City,User,Asgie,Video,Rating

logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s', level=logging.DEBUG)


class RedisConnector(object):

    def __init__(self, url):
        logging.debug("Connecting REDIS DB...")
        self.__redis = redis.StrictRedis.from_url(url)

    def save_video_similarities(self, dictionary_similarities, default_key, separator):
        # update redis db
        logging.debug("Saving content similarities...")
        for video_id, similar_items in dictionary_similarities.items():
            key = "%s%s%s" % (default_key, separator, video_id)
            self.__redis.delete(key)
            for similar_id, date_creation, similar_confidence_level in similar_items:
                # print("content_similarity key={} value={} confidence={}".format(key, similar_id, similar_confidence_level))
                self.__redis.rpush(key, "%s%s%s" % (similar_id, separator, similar_confidence_level))
                # if float(similar_confidence_level) > 0.9:
                #     print("\nid1={} >> id2={} confidence={}\n\n\n".format(similar_id, video_id, similar_confidence_level) )
        logging.debug("Content similarities saved! n=%d..." % len(dictionary_similarities))

    def save_user_recommendations(self, user_id, user_recommendations, default_key, separator):
        key = "%s%s%s" % (default_key, separator, user_id)
        logging.debug("Saving user recommendations user_id={} n_recommendations={} key={} values={}".format(user_id, len(user_recommendations), key, user_recommendations))
        self.__redis.delete(key)
        for item in user_recommendations:
            # item[0] == video_id
            self.__redis.rpush(key, item[0])


class TV4EDataConnector(object):
    __URL_VIDEOS = 'http://api_mysql.tv4e.pt/api/recommendations/videos'
    __URL_RATINGS = 'http://api_mysql.tv4e.pt/api/recommendations/ratings'
    __URL_USERS = 'http://api_mysql.tv4e.pt/api/recommendations/users'
    __NOW = datetime.datetime.now()
    __RATINGS_VALIDITY_IN_DAYS = 14
    __MINIMUM_AMOUNT_OF_VIDEO_TO_CONSIDER_RATED = 10

    def __init__(self, persist_to_db=False):
        self.__persist_to_db = persist_to_db

        self.__dataframe_videos = None
        self.__dataframe_ratings = None
        self.__dataframe_users = None

    def load_users(self):
        """
        Loads the DataFrame with contents. 
        :return: DataFrame with user_id, user_age, user_gender, city_id and user_coordinates
        """
        logging.debug("Loading users data...")

        # loading videos
        data=requests.get(self.__URL_USERS)
        self.__dataframe_users=pd.DataFrame(data.json())

        if self.__persist_to_db:
            for index, row in self.__dataframe_users.iterrows():
                try:
                    if User.objects.filter(pk=row.user_id).count() == 0:
                        user=User(
                            id=row.user_id,
                            age=row.user_age,
                            name=row.user_name,
                            gender=row.user_gender,
                            city=City.objects.only('id').get(id=row.city_id),
                            coordinates=row.user_coordinates,
                        )
                    else:
                        user = User.objects.get(id=row.user_id)
                        user.age=row.user_age
                        user.name=row.user_name
                        user.gender=row.user_gender
                        user.city=City.objects.get(id=row.city_id)
                        user.coordinates=row.user_coordinates
                    user.save()
                except:
                    logging.error("Error while saving User: user_id={}".format(row.user_id))
                    traceback.print_exc()

        logging.debug("Users data loaded! n=%s" % self.__dataframe_users.shape[0])

        return self.__dataframe_users

    def load_videos(self):
        """
        Loads the DataFrame with contents.
        :return: DataFrame with video_id, video_title, video_desc, video_date_creation, video_location,
                 video_asgie_id and video_asgie_title_pt
        """
        logging.debug("Loading videos data...")

        # loading videos
        data=requests.get(self.__URL_VIDEOS)
        self.__dataframe_videos=pd.DataFrame(data.json())
        # XXX transposing as the API returns a pre index list of videos
        # self.__dataframe_videos = self.__dataframe_videos.transpose()
        if self.__persist_to_db:
            for index, row in self.__dataframe_videos.iterrows():
                try:
                    if Video.objects.filter(pk=row.video_id).count() == 0:
                        video=Video(
                            id=row.video_id,
                            title=row.video_title,
                            desc=row.video_desc,
                            date_creation=pytz.utc.localize(dateutil.parser.parse(row.video_date_creation)),
                            location=row.video_location,
                            asgie=Asgie.objects.only('id').get(id=row.video_asgie_id),
                            duration=row.duration
                        )
                    else:
                        video = Video.objects.get(id=row.video_id)
                        video.title=row.video_title
                        video.desc=row.video_desc
                        video.date_creation=pytz.utc.localize(dateutil.parser.parse(row.video_date_creation))
                        video.location=row.video_location
                        video.asgie=Asgie.objects.only('id').get(id=row.video_asgie_id)
                        video.duration=row.duration
                    video.save()
                except:
                    logging.error("Error while saving Video: video_id={}".format(row.video_id))
                    traceback.print_exc()

        self.__dataframe_videos['video_contents'] = self.__dataframe_videos[['video_title', 'video_desc']].\
            apply(lambda x: " ".join(x), axis=1)

        logging.debug("Informative videos data loaded! n=%s" % self.__dataframe_videos.shape[0])

        return self.__dataframe_videos

    def __pre_clean_ratings_data(self):
        """
        Create and calculate implicit/explicit ratings fields
        """
        # XXX ensuring the ratings format
        if self.__dataframe_ratings.empty:
            self.__dataframe_ratings = pd.DataFrame(columns=['user_id', 'video_id', 'video_watch_time', 'rating_date_creation', 'rating_value', 'video_watched_type'])

        # XXX use a function to calculate implicit rating considering the video lead time
        self.__dataframe_ratings['rating_implicit'] = (self.__dataframe_ratings['video_watch_time']/100) * 0.3
        self.__dataframe_ratings['rating_explicit'] = (self.__dataframe_ratings['rating_value'])         * 0.7
        # If the explicit rating was negative, the implicit will be negative
        self.__dataframe_ratings.loc[(self.__dataframe_ratings.rating_explicit < 0), 'rating_implicit'] =                 \
            self.__dataframe_ratings['rating_implicit'] * -1
        # create a new column to put implicit or explicit rating rating_value
        self.__dataframe_ratings['overall_rating_value'] =                                                                \
            self.__dataframe_ratings['rating_implicit'] + self.__dataframe_ratings['rating_explicit']

        # implicit rating is the watched time / explicit rating is the like-0-dislike
        self.__dataframe_ratings['rating_implicit'] = self.__dataframe_ratings['video_watch_time']/100
        self.__dataframe_ratings['rating_explicit'] = self.__dataframe_ratings['rating_value']
        
        # Right now, the overall rating will be NONE/NaN if no explicit rating screen was shown (rating_value is NaN)
        # So, we consider the implicit rating if the user has seen at least a MINIMUM % of the video
        self.__dataframe_ratings.loc[                                                                                     \
            (pd.isnull(self.__dataframe_ratings['rating_value'])) &                                                  \
            (self.__dataframe_ratings['video_watch_time'] >= self.__MINIMUM_AMOUNT_OF_VIDEO_TO_CONSIDER_RATED),           \
                'overall_rating_value'] = (self.__dataframe_ratings['video_watch_time']/100) * 0.5

        # If the explicit rating screen was shown but not answered, we also require this MINIMUM %
        # of time to consider this rating
        self.__dataframe_ratings.loc[
            (pd.isnull(self.__dataframe_ratings['rating_value'])) &  
            (self.__dataframe_ratings['video_watch_time'] < self.__MINIMUM_AMOUNT_OF_VIDEO_TO_CONSIDER_RATED),             \
                'overall_rating_value'] = 0

        # Now, if the explicit rating screen was shown but not answered, we also require this MINIMUM %
        # of time to consider this rating
        self.__dataframe_ratings.loc[(self.__dataframe_ratings['rating_value'] == 0) &                                     \
            (self.__dataframe_ratings['video_watch_time'] < self.__MINIMUM_AMOUNT_OF_VIDEO_TO_CONSIDER_RATED),             \
                'overall_rating_value'] = 0

        # Filling with zeros if the overall_rating is NaN
        self.__dataframe_ratings['overall_rating_value']=self.__dataframe_ratings['overall_rating_value'].fillna(0)

    def __post_clean_ratings_data(self):
        """
        Cutt off rating created before RATINGS_VALIDITY_IN_DAYS rows 
        """
        # Removed forced ratings (created by pressing BACK key on the remote). 
        # These ratings are usually created during the initial demonstration
        self.__dataframe_ratings = self.__dataframe_ratings[(self.__dataframe_ratings.video_watched_type != 'forced')]

        # Create a new column to indicate the difference between the current date and the date of creation of the rating
        self.__dataframe_ratings['rating_date_diff'] =                                                                     \
             self.__NOW - pd.to_datetime(self.__dataframe_ratings['rating_date_creation'])
        # Cutting off ratings created before some days (we don't want ratings created before XX days)
        # Rationale: algorithm was too slow already!!!!
        self.__dataframe_ratings =                                                                                         \
            self.__dataframe_ratings[(self.__dataframe_ratings.rating_date_diff.dt.days < self.__RATINGS_VALIDITY_IN_DAYS)]

        # WE CANNOT IGNORE ZEROED RATINGS, AS IF SO WE MIGHT RECOMMEND S/THING THE USER HAS ALREADY RECEIVED!
        # # Cutting off ZEROed ratings
        # self.__dataframe_ratings =                                                                                         \
        #     self.__dataframe_ratings[(self.__dataframe_ratings.overall_rating_value != 0)]

    def load_ratings(self):
        """
        Loads the DataFrame with contents. 
        :return: DataFrame with user_id, video_id, video_watch_time, rating_date_creation, rating_value,
                 video_watched_type
        """
        logging.debug("Loading ratings data...")

        # loading ratings
        data=requests.get(self.__URL_RATINGS)
        self.__dataframe_ratings=pd.DataFrame(data.json())
        
        self.__pre_clean_ratings_data()

        if self.__persist_to_db:
            Rating.objects.all().delete()
            for index, row in self.__dataframe_ratings.iterrows():
                try:
                    rating=Rating(
                        user=User.objects.only('id').get(id=row.user_id),
                        video=Video.objects.only('id').get(id=row.video_id),
                        watch_time=row.video_watch_time,
                        date_creation=pytz.utc.localize(dateutil.parser.parse(row.rating_date_creation)),
                        watched_type=row.video_watched_type,
                        rating_implicit=row.rating_implicit,
                        overall_rating_value=row.overall_rating_value
                    )
                    if not np.isnan (row.rating_explicit):
                        rating.rating_explicit = row.rating_explicit
                    rating.save()
                except:
                    logging.error("Error while saving Rating: user_id={} video_id={}".format(row.user_id, row.video_id))
                    traceback.print_exc()

        logging.debug("Ratings data loaded! n=%s" % self.__dataframe_ratings.shape[0])

        self.__post_clean_ratings_data()

        logging.debug("Ratings data cleaned! n=%s" % self.__dataframe_ratings.shape[0])

        return self.__dataframe_ratings


