# encoding: utf-8
import logging
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
from math import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.decomposition import TruncatedSVD
import datetime
import dateutil

from majordomo.models import Video

from django.conf import settings
logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s', level=logging.DEBUG)


class NoValidRatingsFoundException(Exception):
    """Exception thrown if no valid ratings are found in the process of calculating user profile"""
    pass


class GeographicFilter(object):

    def __init__(self, dataframe_videos):
        self.__dataframe_videos = dataframe_videos

    def filter(self, city_id):
        logging.debug("Filtering geographically relevant contents city_id={}".format(city_id))
        return \
            self.__dataframe_videos[(self.__dataframe_videos['video_location'] == str(city_id)) |
                                    (self.__dataframe_videos['video_location'] == '')]


class TimeDecayFilter(object):

    def __init__(self, dataframe_videos):
        self.__dataframe_videos = dataframe_videos
        pd.to_datetime(self.__dataframe_videos.video_date_creation)
        self.__dataframe_videos=self.__dataframe_videos.sort_values(['video_date_creation'])
        self.__NOW = datetime.datetime.now()

    def filter(self, user_id, user_recommendations, dataframe_user_ratings, n_recommendations=settings.NUMBER_OF_RECOMMENDATIONS):
        logging.debug("Filtering time decay recommendations for user_id={} n_recommendations={}".format(user_id, n_recommendations))
        # COLD-START handling: If no recommendations are passed, retuns the latest ones 
        if user_recommendations is None:
            # XXX Double checking to avoid already recommended items - if clause down there :)
            filtered_recommendations = [(row.video_id, row.video_date_creation, "")
                                        for index,row in self.__dataframe_videos.iterrows()
                                        if row.video_id not in dataframe_user_ratings.video_id.values][:-n_recommendations-1:-1]
            return filtered_recommendations
        else:
            # Apply time decay algorithm to the list
            filtered_recommendations = []
            for item in user_recommendations:
                timedelta = self.__NOW - dateutil.parser.parse(item[1])
                filtered_recommendations.append((
                    item[0],
                    item[1],
                    item[2]/(1 + timedelta.days)
                ))

            # Order again by the confidence
            return sorted(filtered_recommendations, key=lambda tup: tup[2], reverse=True)[:n_recommendations]


class ContentBasedRecommender(object):

    __NUMBER_OF_ASGIE_TYPES = 7

    def __init__(self, n_similar=settings.NUMBER_OF_RECOMMENDATIONS, dataframe_videos=None):
        self.__n_similar = n_similar
        self.__dataframe_videos = dataframe_videos

        self.__tfidf_vectorizer = None
        self.__tfidf_matrix = None
        self.__tfidf_tokens_dict = None

        self.__NOW = datetime.datetime.now()

    def __create_tfidf_tokens_dict(self):
        # create dict content_id ==>> tfidf weights
        self.__tfidf_tokens_dict = {}
        tfidf_array = self.__tfidf_matrix.toarray()
        line_count = 0
        for idx, row in self.__dataframe_videos.iterrows():
            self.__tfidf_tokens_dict[int(row.video_id)] = tfidf_array[line_count]
            line_count += 1

    def __vectorize(self):
        """
        Vectorize training data, i.e. perform a 2-gram feature extraction and selection using a TF-IDF method 
        :return: Result is a numeric and weighted feature vector notation for each item
        """
        logging.debug("Vectorizing text contents...")
        # nltk.download("stopwords")
        self.__tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, max_df=0.5,
                                                  stop_words=stopwords.words('portuguese'))
        self.__tfidf_matrix = self.__tfidf_vectorizer.fit_transform(self.__dataframe_videos['video_contents'])

        self.__create_tfidf_tokens_dict()

        logging.debug("Content features loaded! n=%s" % len(self.__tfidf_vectorizer.vocabulary_))

    def save_video_tokens(self):
        """
        Store video tokens in databadse
        """
        if self.__tfidf_matrix is None:
            self.__vectorize()

        logging.debug("Saving video tokens...")

        vectors = self.__tfidf_matrix.toarray()
        i = 0
        for video_id, row in self.__dataframe_videos.iterrows():
            video = Video.objects.get(pk=row.video_id)
            video.tokens = ", ".join(self.__tfidf_vectorizer.inverse_transform(vectors[i])[0])
            i = i+1
            video.save()

    def find_similarities(self):
        """
        Find the n most similar items for each item in the DataFrame
        :return: Dictionary of video_id : [similar_item1 .. similar_item_n]
        """
        if self.__tfidf_matrix is None:
            self.__vectorize()

        logging.debug("Calculating content similarities...")
        n_similar = int((self.__n_similar + 2))
        cosine_similarities = linear_kernel(self.__tfidf_matrix, self.__tfidf_matrix)

        dictionary_similarities = {}
        i = 0
        for index, row in self.__dataframe_videos.iterrows():
            # print("\nrow.video_id = %s" % row.video_id)
            # print("    row.video_title: %s" % row.video_title)
            similar_indices = cosine_similarities[i].argsort()[:-n_similar:-1]
            similar_indices = similar_indices[1:]
            similar_items = [(self.__dataframe_videos.iloc[j].video_id,
                              self.__dataframe_videos.iloc[j].video_date_creation,
                              cosine_similarities[i][j]) for j in similar_indices]
            # print("    similar_items = %s" % similar_items)

            dictionary_similarities[row.video_id] = similar_items
            i = i + 1

        return dictionary_similarities

    def visualize_data(self):
        if self.__tfidf_matrix is None:
            self.__vectorize()

        logging.debug("Preparing visualization of data...")
        # Fit and transform data to n_features-dimensional space
        svd = TruncatedSVD()
        tfidf_matrix_reduced = svd.fit_transform(self.__tfidf_matrix)
        # create a new column for the coordinates
        self.__dataframe_videos['x_coordinate'] = range(0, len(tfidf_matrix_reduced))
        self.__dataframe_videos['x_coordinate'] = self.__dataframe_videos.x_coordinate.apply(lambda index: tfidf_matrix_reduced[index,0 :1])
        self.__dataframe_videos['y_coordinate'] = range(0, len(tfidf_matrix_reduced))
        self.__dataframe_videos['y_coordinate'] = self.__dataframe_videos.y_coordinate.apply(lambda index: tfidf_matrix_reduced[index,1 :])
        # prepare markers (we know we have 7 ASGIE types, so we'll set 7 markers)
        n_asgie_title_pt = len(self.__dataframe_videos['video_asgie_title_pt'].unique())
        markers_choice_list = ['o', 's', '^', '.', 'v', '<', '>']
        markers_list = [markers_choice_list[i % self.__NUMBER_OF_ASGIE_TYPES] for i in range(n_asgie_title_pt)]
        # plot!
        sns.lmplot("x_coordinate", "y_coordinate", hue="video_asgie_title_pt", data=self.__dataframe_videos, fit_reg=False, markers=markers_list, scatter_kws={"s": 150})
        # Adjust borders and add title
        sns.set(font_scale=2)
        plt.title('Visualization of +TV4E Informative Videos in a 2-dimensional space')
        plt.subplots_adjust(right=0.80, top=0.90, left=0.12, bottom=0.12)
        # Show plot
        plt.show()

    def __calculate_user_profile(self, user_id, user_ratings):
        # created weighted user profile vector (dotproduct of vectors of items consumed and user ratings)
        #   "In the original implementation, the profile was the sum of the item-tag vectors of all items 
        #   the user has rated positively (>= 3.5 stars). This approach was later improved with weighted 
        #   user profile (with the older implementation commented out for reference). Weighted profile is 
        #   computed with weighted sum of the item vectors for all items, with weights being based on the 
        #   user's rating."
        #   See: http://eugenelin89.github.io/recommender_content_based/
        if (user_ratings.shape[0] == 0):
            raise NoValidRatingsFoundException

        user_profile = [0] * len(self.__tfidf_vectorizer.get_feature_names())
        logging.debug("Calculating content-based user profile for user_id={} n_ratings={}".format(user_id, user_ratings.shape[0]))
        for idx, row in user_ratings.iterrows():

            if row.overall_rating_value == 0:
                continue
            weighted_rating = float(row.overall_rating_value/(row.rating_date_diff.days + 1))

            for i in range(len(user_profile)):
                # Apply time decay to ratings also!
                # timedelta = self.__NOW - dateutil.parser.parse(row.rating_date_creation)
                # weighted_rating = float(row.overall_rating_value/(timedelta.days + 1))
                try:
                    weight_of_word = float(self.__tfidf_tokens_dict[row.video_id][i])
                except:
                    logging.error("Token not found! video_id={} user_id={}".format(row.video_id, user_id))
                    break
                user_profile[i] += weighted_rating * weight_of_word
                # user_profile = [v/len(user_ratings) for v in user_profile] # weight-ing user vector (?)
        # normalize user profile vector
        normal_factor = np.linalg.norm(user_profile)
        if normal_factor == 0:
            raise NoValidRatingsFoundException
        
        user_profile = user_profile / normal_factor

        return user_profile

    def calculate_recommendations(self, user_id, dataframe_user_ratings):
        # apply cosine similarity between user profile vector and content vectors
        # See: http://eugenelin89.github.io/recommender_content_based/
        logging.debug("Calculating content-based recommendations for user_id={} n_similar={} n_ratings={}".
              format(user_id, self.__n_similar, dataframe_user_ratings.shape[0]))
        try:
            user_profile = self.__calculate_user_profile(user_id, dataframe_user_ratings)
        except NoValidRatingsFoundException:
            logging.warning("No valid ratings found for user_id={}".format(user_id))
            return None

        n_similar = (self.__n_similar + 1)
        # calculate similarity using cosine
        user_recommendations = []
        for video_id, token_weights in self.__tfidf_tokens_dict.items():
            # not calculating for contents already consumed
            if video_id not in dataframe_user_ratings.video_id.values:
                user_recommendations.append((
                    video_id,
                    self.__dataframe_videos[(self.__dataframe_videos.video_id == video_id)].video_date_creation.values[0],
                    cosine_similarity([user_profile], [token_weights])[0]
                ))
        # order ratings by similarity
        user_recommendations = sorted(user_recommendations, key=lambda tup: tup[2], reverse=True)[:n_similar]

        return user_recommendations
