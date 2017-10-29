from recommenders import ContentBasedRecommendations
from data import LocalRedisConnector, TV4EDataConnector

if __name__ == "__main__":

    tv4e_connector = TV4EDataConnector()
    dataframe_videos = tv4e_connector.load_videos()
    dataframe_ratings = tv4e_connector.load_ratings()

    content_based_rec = ContentBasedRecommendations(dataframe_videos=dataframe_videos)
    dictionary_similarities = content_based_rec.find_similarities()
    LocalRedisConnector().save_video_similarities(dictionary_similarities)
    #content_based_rec.visualize_data()

