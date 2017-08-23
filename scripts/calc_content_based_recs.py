# encoding: utf-8
import logging
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
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
import redis
import operator
from math import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD	

from view.models import Asgie, InformativeVideos, AsgieAvResource
from rs.models import Senior, Rating

logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s', level=logging.DEBUG)


class ContentBasedRecommendations(object):

	__OUTPUT_FILENAME = 'output.csv'
	__RAW_DATA_FILENAME = 'informative_videos_raw_data.csv'
	__NUMBER_OF_ASGIE_TYPES = 7

	def __init__(self, n_similar=10, consider_asgie_categories=False, save_raw_data_to_csv=False):
		self.__n_similar = n_similar
		self.__consider_asgie_categories = consider_asgie_categories
		self.__save_raw_data_to_csv = save_raw_data_to_csv

		self.__dataframe_videos = None
		self.__dataframe_ratings = None
		self.__tfidf_vectorizer = None
		self.__tfidf_matrix = None
		self.__tfidf_tokens_dict = None
		self.__redis = None


	def __connect_redis(self):
		logging.debug("Connecting REDIS DB...")
		self.__redis = redis.StrictRedis.from_url(settings.REDIS_URL)

	def __load_data(self):
		"""
		Loads the DataFrame with contents. 
		:return: DataFrame with id, title and desc of items
		"""
		logging.debug("Loading data...")
		# loading videos
		videos = InformativeVideos.objects.all()
		data = np.array([[video.id, video.title, video.desc, video.asgie_title_pt] for video in videos])
		self.__dataframe_videos = pd.DataFrame(data=data[0:,0:], index=data[0:,0], columns=['id', 'title', 'desc', 'asgie_title_pt'])
		if self.__save_raw_data_to_csv:
		    logging.debug("Saving raw data to CSV [%s..." % self.__RAW_DATA_FILENAME)
		    self.__dataframe_videos.to_csv(self.__RAW_DATA_FILENAME, encoding='utf-8', sep=',', index=False)
		self.__dataframe_videos['text_contents'] = self.__dataframe_videos[['title', 'desc']].apply(lambda x: " ".join(x), axis=1)
	  #   if self.__consider_asgie_categories:
			# df_asgie = pd.DataFrame(None)
			# df_asgie['asgie_title_pt'] = self.__dataframe_videos['asgie_title_pt']
			# dummies = pd.get_dummies(df_asgie).astype(int)
			# self.__dataframe_videos = np.concatenate((self.X_text_contents, dummies), axis=1)
		# loading ratings
		ratings = Rating.objects.all()
		data = np.array([[rating.user_id, rating.content_id, rating.rating] for rating in ratings])
		self.__dataframe_ratings = pd.DataFrame(data=data[0:,0:], index=data[0:,0], columns=['user_id', 'content_id', 'rating'])
		self.__dataframe_ratings =  self.__dataframe_ratings.apply(pd.to_numeric) # XXX necesssary in Python 2.7

		logging.debug("Informative videos data loaded! n=%s" % self.__dataframe_videos.shape[0])
		logging.debug("Ratings data loaded! n=%s" % self.__dataframe_ratings.shape[0])


	def __vectorize(self):
		"""
		Vectorize training data, i.e. perform a 2-gram feature extraction and selection using a TF-IDF method 
		:return: Result is a numeric and weighted feature vector notation for each item
		"""
		logging.debug("Vectorizing text contents...")
		self.__tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, max_df=0.5, stop_words=stopwords.words('portuguese'))
		self.__tfidf_matrix = self.__tfidf_vectorizer.fit_transform(self.__dataframe_videos['text_contents'])

		logging.debug("Content features loaded! n=%s" % len(self.__tfidf_vectorizer.vocabulary_))


	def __create_tfidf_tokens_dict(self):
		# create dict content_id ==>> tfidf weights
		self.__tfidf_tokens_dict = {}
		tfidf_array = self.__tfidf_matrix.toarray()
		line_count = 0
		for idx, row in self.__dataframe_videos.iterrows():
			self.__tfidf_tokens_dict[int(row.id)] = tfidf_array[line_count]
			line_count += 1


	def __calculate_user_profile(self, user_id):
		# created weighted user profile vector (dotproduct of vectors of items consumed and user ratings)
		#   "In the original implementation, the profile was the sum of the item-tag vectors of all items 
		#   the user has rated positively (>= 3.5 stars). This approach was later improved with weighted 
		#   user profile (with the older implementation commented out for reference). Weighted profile is 
		#   computed with weighted sum of the item vectors for all items, with weights being based on the 
		#   user's rating."
		#   See: http://eugenelin89.github.io/recommender_content_based/
		user_profile = [0] * len(self.__tfidf_vectorizer.get_feature_names())
		user_ratings = self.__dataframe_ratings[self.__dataframe_ratings.user_id==user_id]
		logging.debug("Calculating user profile for user id=%s n_ratings=%s..." % (user_id, user_ratings.shape[0]))
		for i in range(len(user_profile)):
			for idx, row in self.__dataframe_ratings.iterrows():
				#print('i = %s rating = %s content_id = %s' % (i, row.rating, row.content_id))
				user_profile[i] += row.rating * self.__tfidf_tokens_dict[row.content_id][i]
			#user_profile = [v/len(user_ratings) for v in user_profile] # weight-ing user vector (?)
		# normalize user profile vector
		user_profile = user_profile / np.linalg.norm(user_profile)
		return user_profile


	def __cosine_similarity(self, x, y):
		def square_rooted(v):
			return round(sqrt(sum([a * a for a in v])), 3)

		numerator = sum(a * b for a, b in zip(x, y))
		denominator = square_rooted(x) * square_rooted(y)
		return numerator/float(denominator)


	def __calculate_recommendations(self):
		# apply cosine similarity between user profile vector and content vectors
		# See: http://eugenelin89.github.io/recommender_content_based/
		n_similar = (self.__n_similar + 1)
		users_id = self.__dataframe_ratings.user_id.unique()
		for user_id in users_id:
			user_profile = self.__calculate_user_profile(user_id)
    		# calculate similarity using cosine
			estimated_user_ratings = {}
			for content_id, token_weights in self.__tfidf_tokens_dict.iteritems(): 
				if content_id not in self.__dataframe_ratings.content_id.values: # not calculating for contents already consumed 
					estimated_user_ratings[content_id] = self.__cosine_similarity(user_profile, token_weights)
			# order ratings
			estimated_user_ratings = sorted(estimated_user_ratings.items(), key=operator.itemgetter(1))[:-n_similar:-1]
			logging.debug("Saving recommendations for user id=%s n_recs=%s..." % (user_id, self.__n_similar))

			# update redis db
			key = "%s%s%s" % (settings.KEY_USER_RECOMMENDATION, settings.SEPARATOR, user_id)
			self.__redis.delete(key)
			for content_id, similarity in estimated_user_ratings:
				self.__redis.rpush(key, content_id)


	def calculate_and_save_recommendations(self):
		"""
		Load and transform the +TV4E informative contents, train a content-based recommender system and make a recommendation for each video
		:return:
		"""
		self.__connect_redis()
		self.__load_data()
		self.__vectorize()
		self.__create_tfidf_tokens_dict()
		self.__calculate_recommendations()



if __name__ == "__main__":
	proc = ContentBasedRecommendations()
	proc.calculate_and_save_recommendations()

