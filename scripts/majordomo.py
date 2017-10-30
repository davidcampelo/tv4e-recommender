from recommenders import ContentBasedRecommender, GeographicRecommender
from data import LocalRedisConnector, TV4EDataConnector

if __name__ == "__main__":

    tv4e_connector = TV4EDataConnector()
    dataframe_videos = tv4e_connector.load_videos()
    dataframe_ratings = tv4e_connector.load_ratings()
    dataframe_users = tv4e_connector.load_users()

    # content_based_rec = ContentBasedRecommendations(dataframe_videos=dataframe_videos)
    # dictionary_similarities = content_based_rec.find_similarities()
    # LocalRedisConnector().save_video_similarities(dictionary_similarities)

    locations = dataframe_users.city_id.unique()
    for location_id in locations:
        # Filter geographically relevant videos
        geographic_rec = GeographicRecommender(dataframe_videos=dataframe_videos)
        dataframe_videos_filtered = geographic_rec.filter(location_id)

        # Creating a content-based recommender
        content_based_rec = ContentBasedRecommender(dataframe_videos=dataframe_videos_filtered)
        content_based_rec.find_similarities()

        # Calculating user recommendations
        for index, user in dataframe_users[dataframe_users.city_id == location_id].iterrows():
            user_id = user.user_id
            user_ratings = dataframe_ratings[dataframe_ratings.user_id == user_id]
            user_recommendations = content_based_rec.calculate_recommendations(user_id, user_ratings)



#    content_based_rec.visualize_data()

