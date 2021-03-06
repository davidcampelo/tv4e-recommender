# encoding: utf-8
import logging
import pandas as pd
import requests
import redis

logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s', level=logging.DEBUG)


class RedisConnector(object):

    def __init__(self, url):
        logging.debug("Connecting REDIS DB...")
        self.__redis = redis.StrictRedis.from_url(url)

    def save_video_similarities(self, dictionary_similarities, key, separator):
        # update redis db
        logging.debug("Saving content similarities...")
        for video_id, similar_items in dictionary_similarities.items():
            key = "%s%s%s" % (key, separator, video_id)
            self.__redis.delete(key)
            for similar_id, similar_confidence_level in similar_items:
                # print("content_similarity: %s =>> %s (%s)" % (video_id, similar_id, similar_confidence_level))
                self.__redis.rpush(key, "%s%s%s" % (similar_id, separator, similar_confidence_level))
                # if float(similar_confidence_level) > 0.9:
                    # print("\nid1={} >> id2={} confidence={}\n\n\n".format(similar_id, video_id, similar_confidence_level) )
        logging.debug("Content similarities saved! n=%d..." % len(dictionary_similarities))

    def save_user_recommendations(self, user_id, user_recommendations, key, separator):
        key = "%s%s%s" % (key, separator, user_id)
        logging.debug("Saving user recomendations user_id={} n_recommendations={} key={}".format(user_id, len(user_recommendations), key))
        self.__redis.delete(key)
        for item in user_recommendations:
            # item[0] == video_id
            self.__redis.rpush(key, item[0])


class TV4EDataConnector(object):
    __URL_VIDEOS = 'http://api_mysql.tv4e.pt/api/recommendations/videos'
    __URL_RATINGS = 'http://api_mysql.tv4e.pt/api/recommendations/ratings'
    __URL_USERS = 'http://api_mysql.tv4e.pt/api/recommendations/users'

    def __init__(self, save_raw_data_to_csv=False):
        self.__save_raw_data_to_csv = save_raw_data_to_csv

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
        self.__dataframe_videos = self.__dataframe_videos.transpose()
        if self.__save_raw_data_to_csv:
            logging.debug("Saving raw data to CSV [%s..." % self.__RAW_DATA_FILENAME)
            self.__dataframe_videos.to_csv(self.__RAW_DATA_FILENAME, encoding='utf-8', sep=',', index=False)
        self.__dataframe_videos['video_contents'] = self.__dataframe_videos[['video_title', 'video_desc']].\
            apply(lambda x: " ".join(x), axis=1)

        logging.debug("Informative videos data loaded! n=%s" % self.__dataframe_videos.shape[0])

        return self.__dataframe_videos

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
        # calculate implicit and explicit ratings
        # XXX use a function to calculate implicit rating considering the video lead time
        self.__dataframe_ratings['rating_implicit'] = (self.__dataframe_ratings['video_watch_time']/100) * 0.3
        self.__dataframe_ratings['rating_explicit'] = (self.__dataframe_ratings['rating_value'])         * 0.7

        # create a new column to put implicit or explicit rating value
        self.__dataframe_ratings['overall_rating_value'] = self.__dataframe_ratings['rating_implicit'] + self.__dataframe_ratings['rating_explicit']

        logging.debug("Ratings data loaded! n=%s" % self.__dataframe_ratings.shape[0])

        return self.__dataframe_ratings


