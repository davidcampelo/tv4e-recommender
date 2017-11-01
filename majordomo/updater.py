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

from recommenders import ContentBasedRecommender, GeographicFilter, TimeDecayFilter
from data import TV4EDataConnector, RedisConnector


class Updater(object):
    @staticmethod
    def update_tv4e_data():
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

        content_based_rec.visualize_data()

    @staticmethod
    def update_recommendations():
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
            content_based_rec = ContentBasedRecommender(n_similar=settings.NUMBER_OF_RECOMMENDATIONS*2,
                                                        dataframe_videos=dataframe_videos_filtered)
            content_based_rec.find_similarities()
            time_filter = TimeDecayFilter(dataframe_videos=dataframe_videos_filtered)

            # Calculating user recommendations
            for index, user in dataframe_users[dataframe_users.city_id == location_id].iterrows():
                user_id = user.user_id
                dataframe_user_ratings = dataframe_ratings[dataframe_ratings.user_id == user_id]
                # If the user has at least one rating
                user_recommendations = content_based_rec.calculate_recommendations(user_id, dataframe_user_ratings)

                # Apply the time decay
                user_recommendations = time_filter.filter(n_recommendations=settings.NUMBER_OF_RECOMMENDATIONS,
                                                          user_id=user_id,
                                                          user_recommendations=user_recommendations)

                # XXX Save it!
                redis.save_user_recommendations(user_id=user_id,
                                                default_key=settings.KEY_USER_RECOMMENDATION,
                                                separator=settings.SEPARATOR,
                                                user_recommendations=user_recommendations)


if __name__ == "__main__":
    Updater().update_tv4e_data()
    Updater().update_recommendations()
