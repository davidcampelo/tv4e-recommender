import django
# XXX work around to run this script in a sub-directory of the project (/scripts)
import os
import sys

root_path = os.path.abspath(os.path.split(__file__)[0])
sys.path.insert(0, os.path.join(root_path, '../'))
sys.path.insert(0, root_path)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tv4e.settings")
django.setup()

from django.conf import settings
import logging
import traceback

from recommenders import ContentBasedRecommender, GeographicFilter, TimeDecayFilter
from data import TV4EDataConnector, RedisConnector
from lock import LockedModel, AlreadyLockedError

logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s', level=logging.DEBUG)

class Updater(LockedModel):
    id = "0"

    def update_tv4e_data(self):
        logging.info("***** update_tv4e_data() IN")
        redis = RedisConnector(url=settings.REDIS_URL)
        tv4e_connector = TV4EDataConnector(persist_to_db=True)
        dataframe_videos = tv4e_connector.load_videos()
        tv4e_connector.load_users()
        tv4e_connector.load_ratings()

        content_based_rec = ContentBasedRecommender(n_similar=settings.NUMBER_OF_RECOMMENDATIONS,
                                                    dataframe_videos=dataframe_videos)
        content_based_rec.save_video_tokens()
        dictionary_similarities = content_based_rec.find_similarities()
        redis.save_video_similarities(
            dictionary_similarities=dictionary_similarities,
            default_key=settings.KEY_CONTENT_SIMILARITY,
            separator=settings.SEPARATOR
        )

        logging.info("***** update_tv4e_data() OUT")
        #content_based_rec.visualize_data()


    def update_recommendations(self):

        logging.info("***** update_recommendations() IN")
        redis = RedisConnector(url=settings.REDIS_URL)
        tv4e_connector = TV4EDataConnector()
        dataframe_videos = tv4e_connector.load_videos()
        dataframe_ratings = tv4e_connector.load_ratings()
        dataframe_users = tv4e_connector.load_users()

        locations = dataframe_users.city_id.unique()
        for location_id in locations:
            # Filter geographically relevant videos
            geo_filter = GeographicFilter(dataframe_videos=dataframe_videos)
            dataframe_videos_filtered = geo_filter.filter(location_id)

            # Creating a content-based recommender
            content_based_rec = ContentBasedRecommender(n_similar=settings.NUMBER_OF_RECOMMENDATIONS*3,
                                                        dataframe_videos=dataframe_videos_filtered)
            content_based_rec.find_similarities()
            time_filter = TimeDecayFilter(dataframe_videos=dataframe_videos_filtered)

            # Calculating user recommendations
            for index, user in dataframe_users[dataframe_users.city_id == location_id].iterrows():
                user_id = user.user_id
                dataframe_user_ratings = dataframe_ratings[dataframe_ratings.user_id == user_id]
                # If the user has at least one rating
                user_recommendations = content_based_rec.calculate_recommendations(user_id=user_id, 
                                                                                   dataframe_user_ratings=dataframe_user_ratings)

                # Apply the time decay
                user_recommendations = time_filter.filter(n_recommendations=settings.NUMBER_OF_RECOMMENDATIONS,
                                                          user_id=user_id,
                                                          dataframe_user_ratings=dataframe_user_ratings,
                                                          user_recommendations=user_recommendations)

                # XXX Save it!
                redis.save_user_recommendations(user_id=user_id,
                                                default_key=settings.KEY_USER_RECOMMENDATION,
                                                separator=settings.SEPARATOR,
                                                user_recommendations=user_recommendations)
        logging.info("***** update_recommendations() OUT")


if __name__ == "__main__":
    updater = Updater()
    error = False
    try:
        updater.lock()
        updater.update_tv4e_data()
        updater.update_recommendations()
    except AlreadyLockedError as err:
        error = True
        logging.warning("***** Called refresh during refresh recommendations!")
    except Exception as err:
        logging.error("***** Unhandled error during refresh_recommendations: {}". format(err))
        traceback.print_exc()
    finally:
        if not error:
            updater.unlock()