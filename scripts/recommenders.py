# encoding: utf-8
import logging
import pandas as pd
import numpy as np
import requests
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
import operator
from math import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD    

logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s', level=logging.DEBUG)

class GeographicRecommender(object):

    def __init__(self, dataframe_videos):
        self.__dataframe_videos = dataframe_videos

    def filter(self, city_id):
        return \
            self.__dataframe_videos[(self.__dataframe_videos['video_location'] == city_id) |
                                    (self.__dataframe_videos['video_location'] == '' )]

class ContentBasedRecommender(object):

    __NUMBER_OF_ASGIE_TYPES = 7

    def __init__(self, n_similar=10, dataframe_videos):
        self.__n_similar = n_similar
        self.__dataframe_videos = dataframe_videos

        self.__tfidf_vectorizer = None
        self.__tfidf_matrix = None
        self.__tfidf_tokens_dict = None


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
        self.__tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, max_df=0.5,
                                                  stop_words=stopwords.words('portuguese'))
        self.__tfidf_matrix = self.__tfidf_vectorizer.fit_transform(self.__dataframe_videos['video_contents'])

        self.__create_tfidf_tokens_dict()

        logging.debug("Content features loaded! n=%s" % len(self.__tfidf_vectorizer.vocabulary_))

    def find_similarities(self):
        """
        Find the n most similar items for each item in the DataFrame
        :return:
        """
        if self.__tfidf_matrix is None:
            self.__vectorize()

        logging.debug("Calculating content similarities...")
        n_similar = int((self.__n_similar + 2) * -1)
        cosine_similarities = linear_kernel(self.__tfidf_matrix, self.__tfidf_matrix)

        dictionary_similarities = {}
        i = 0
        for index, row in self.__dataframe_videos.iterrows():
            print("\nrow.video_id = %s" % row.video_id)
            print("    row.video_title: %s" % row.video_title)
            similar_indices = cosine_similarities[i].argsort()[:-5:-1]
            similar_indices = similar_indices[1:]
            similar_items = [(self.__dataframe_videos.iloc[j].video_id, cosine_similarities[i][j]) for j in similar_indices]
            print("    similar_items = %s" % similar_items)

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
        user_profile = [0] * len(self.__tfidf_vectorizer.get_feature_names())
        logging.debug("Calculating user profile for user id=%s n_ratings=%s..." % (user_id, user_ratings.shape[0]))
        for i in range(len(user_profile)):
            for idx, row in user_ratings.iterrows():
                # print('i = %s rating = %s video_id = %s' % (i, row.overall_rating_value, row.video_id))
                # print('tokens = %s' % self.__tfidf_tokens_dict[row.video_id])
                user_profile[i] += row.overall_rating_value * self.__tfidf_tokens_dict[row.video_id][i]
            # user_profile = [v/len(user_ratings) for v in user_profile] # weight-ing user vector (?)
        # normalize user profile vector
        user_profile = user_profile / np.linalg.norm(user_profile)
        return user_profile

    @staticmethod
    def __cosine_similarity(x, y):
        def square_rooted(v):
            return round(sqrt(sum([a * a for a in v])), 3)

        numerator = sum(a * b for a, b in zip(x, y))
        denominator = square_rooted(x) * square_rooted(y)
        return numerator/float(denominator)

    def calculate_recommendations(self, user_id, user_ratings):
        # apply cosine similarity between user profile vector and content vectors
        # See: http://eugenelin89.github.io/recommender_content_based/
        n_similar = (self.__n_similar + 1)
        user_profile = self.__calculate_user_profile(user_id, user_ratings)
        # calculate similarity using cosine
        estimated_user_ratings = {}
        for video_id, token_weights in self.__tfidf_tokens_dict.items():
            if video_id not in user_ratings.video_id.values: # not calculating for contents already consumed
                estimated_user_ratings[video_id] = self.__cosine_similarity(user_profile, token_weights)
        # order ratings
        estimated_user_ratings = sorted(estimated_user_ratings.items(), key=operator.itemgetter(1))[:-n_similar:-1]
