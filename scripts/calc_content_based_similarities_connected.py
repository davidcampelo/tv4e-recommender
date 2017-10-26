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
import requests
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
import redis
from math import *
from django_pandas.io import read_frame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD	

from view.models import Asgie, InformativeVideos, AsgieAvResource
from rs.models import Senior, Rating, VideoTokens

logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s', level=logging.DEBUG)


class OfflineContentBasedSimilarity(object):

	__OUTPUT_FILENAME = 'output.csv'
	__RAW_DATA_FILENAME = 'informative_videos_raw_data.csv'
	__NUMBER_OF_ASGIE_TYPES = 7
	__URL='http://api_mysql.tv4e.pt/api/recommendations/videos'


	def __init__(self, n_similar=10, consider_asgie_categories=False, save_raw_data_to_csv=False):
		self.__n_similar = n_similar
		self.__consider_asgie_categories = consider_asgie_categories
		self.__save_raw_data_to_csv = save_raw_data_to_csv

		self.__dataframe = None
		self.__tfidf_matrix = None
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
		data=requests.get(self.__URL)
		self.__dataframe=pd.DataFrame(data.json())

		if self.__save_raw_data_to_csv:
		    logging.debug("Saving raw data to CSV [%s]..." % self.__RAW_DATA_FILENAME)
		    self.__dataframe.to_csv(self.__RAW_DATA_FILENAME, encoding='utf-8', sep=',', index=False)
		self.__dataframe['video_contents'] = self.__dataframe[['video_title', 'video_desc']].apply(lambda x: " ".join(x), axis=1)
	  #   if self.__consider_asgie_categories:
			# df_asgie = pd.DataFrame(None)
			# df_asgie['video_asgie_title_pt'] = self.__dataframe['video_asgie_title_pt']
			# dummies = pd.get_dummies(df_asgie).astype(int)
			# self.__dataframe = np.concatenate((self.X_text_contents, dummies), axis=1)

		logging.debug("Data Loaded! Number of items: {0}".format(len(self.__dataframe)))


	def __vectorize(self):
		"""
		Vectorize training data, i.e. perform a 2-gram feature extraction and selection using a TF-IDF method 
		:return: Result is a numeric and weighted feature vector notation for each item
		"""
		logging.debug("Vectorizing text contents...")
		tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, max_df=0.5, stop_words=stopwords.words('portuguese'))
		self.__tfidf_matrix = tfidf.fit_transform(self.__dataframe['video_contents'])
		vectors = self.__tfidf_matrix.toarray()

		i = 0		
		for video_id, row in self.__dataframe.iterrows():
			tokens = ", ".join(tfidf.inverse_transform(vectors[i])[0])
			video_id = row['video_id']
			i += 1

			# videotokens = VideoTokens.objects.filter(video_id=video_id)
			# if videotokens.count() == 0:
			# 	videotokens = VideoTokens(video_id=video_id, tokens=tokens)
			# 	videotokens.save()
			# else:
			# 	videotokens[0].tokens = tokens
			# 	videotokens[0].save()

		logging.debug("Number of features found: %s" % len(tfidf.vocabulary_))


	def __find_and_save_similarities(self):
		"""
		Find the n most similar items for each item in the DataFrame
		:return:
		"""
		logging.debug("Finding similarities...")
		n_similar = int((self.__n_similar + 2) * -1)
		cosine_similarities = linear_kernel(self.__tfidf_matrix, self.__tfidf_matrix)

		i = 0
		for video_id, row in self.__dataframe.iterrows():
			similar_indices = cosine_similarities[i]
			similar_indices = similar_indices.argsort()[:n_similar:-1]
			similar_indices = similar_indices[1:] # the most similar is the item itself, remove it!
			similar_items = [(self.__dataframe['video_id'][j], cosine_similarities[i][j]) for j in similar_indices]
			# update redis db
			key = "%s%s%s" % (settings.KEY_CONTENT_SIMILARITY, settings.SEPARATOR, row['video_id'])
			self.__redis.delete(key)
			for similar_id, similar_confidence_level in similar_items:
				self.__redis.rpush(key, "%s%s%s" % (similar_id, settings.SEPARATOR, similar_confidence_level))
				# print("content_similarity: %s =>> %s (%s)" % (row['id'], similar_id, similar_confidence_level))
			i = i + 1

		logging.debug("A total of [%s] similar contents where saved for [%s] items!" % (self.__n_similar, len(cosine_similarities)))


	def calculate_and_save_similarities(self):
		"""
		Load and transform the +TV4E informative contents, train a content-based recommender system and make a recommendation for each video
		:return:
		"""
		self.__connect_redis()
		self.__load_data()
		self.__vectorize() 
		self.__find_and_save_similarities()


	def visualize_data(self):
		logging.debug("Preparing visualization of data...")
		# Fit and transform data to n_features-dimensional space
		svd = TruncatedSVD()
		self.__tfidf_matrix_reduced = svd.fit_transform(self.__tfidf_matrix)
		# create a new column for the coordinates
		self.__dataframe['x_coordinate'] = range(0, len(self.__tfidf_matrix_reduced))
		self.__dataframe['x_coordinate'] = self.__dataframe.x_coordinate.apply(lambda index: self.__tfidf_matrix_reduced[index,0 :1])
		self.__dataframe['y_coordinate'] = range(0, len(self.__tfidf_matrix_reduced))
		self.__dataframe['y_coordinate'] = self.__dataframe.y_coordinate.apply(lambda index: self.__tfidf_matrix_reduced[index,1 :])
		# prepare markers (we know we have 7 ASGIE types, so we'll set 7 markers)
		n_asgie_title_pt = len(self.__dataframe['video_asgie_title_pt'].unique())
		markers_choice_list = ['o', 's', '^', '.', 'v', '<', '>']
		markers_list = [markers_choice_list[i % self.__NUMBER_OF_ASGIE_TYPES] for i in range(n_asgie_title_pt)]
		# plot!
		sns.lmplot("x_coordinate", "y_coordinate", hue="video_asgie_title_pt", data=self.__dataframe, fit_reg=False, markers=markers_list, scatter_kws={"s": 150})
		# Adjust borders and add title
		sns.set(font_scale=2)
		plt.title('Visualization of +TV4E Informative Videos in a 2-dimensional space')
		plt.subplots_adjust(right=0.80, top=0.90, left=0.12, bottom=0.12)
		# Show plot
		plt.show()


if __name__ == "__main__":
	proc = OfflineContentBasedSimilarity()
	proc.calculate_and_save_similarities()
	proc.visualize_data()
