# encoding: utf-8
import os
import logging

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tv4e.settings")
django.setup()

from django_pandas.io import read_frame
import pandas as pd
import numpy as np
from math import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import PCA, TruncatedSVD	
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

import redis

from view.models import Asgie, InformativeVideos, AsgieAvResource

# XXX Code to remove duplicated videos from DB
# array = [1]
# while len(array) > 0:
#     from django.db.models import Count
#     array = []
#     duplicate = InformativeVideos.objects.values('title').annotate(title_count=Count('title')).filter(title_count__gt=1)
#     for data in duplicate:
#         email = data['title']
#         array.append(InformativeVideos.objects.filter(title=email).order_by('pk')[:1])
#     for video in array:
#         v = InformativeVideos.objects.filter(pk=video)
#         v.delete()


logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s', level=logging.DEBUG)


class ContentBasedRecommender:

    def __init__(self):

        # The values below can be changed to tweak the recommender algorithm
        self.n_features_total = 100
        self.n_most_similar = 4

        self.consider_asgie_categories = False
        self.save_raw_data_to_csv = False
        self.save_raw_data_filename = "videos.csv"
        self.df = None
        self.df_vectors = None
        self.similarity_score_dict = {}
        self.X = None
        self.X_tfidf_matrix = None
        self.X_text_contents = None


    def run(self):
        """
        Load and transform the +TV4E informative contents, train a content-based recommender system and make a recommendation for each video
        :return:
        """
        #self.configure_redis()
        self.load_data()
        self.vectorize()
        if self.consider_asgie_categories:
			df_asgie = pd.DataFrame(None)
			df_asgie['asgie_title_pt'] = self.df['asgie_title_pt']
			dummies = pd.get_dummies(df_asgie).astype(int)
			self.X = np.concatenate((self.X_text_contents, dummies), axis=1)
        else:
			self.X = self.X_text_contents

        #self.X = self.reduce_dimensionality(self.X, n_features=self.n_features_total)
        self.find_similar()
        self.visualize_data()
        self.save_output_to_csv()

    def configure_redis(self):
        # XXX Put url in a config file!
        logging.debug("Flushing REDIS DB...")
        self._redis = redis.StrictRedis.from_url('redis://localhost:6379')
        self._redis.flushdb()


    # Load data
    def load_data(self):
        """
        Loads the DataFrame with contents. 
        :return: DataFrame with id, title and desc of items
        """
        logging.debug("Loading data...")
        videos = InformativeVideos.objects.all()
        # XXX merge title and desc in the for loop
        data = np.array([[video.id, video.title, video.desc, video.asgie_title_pt] for video in videos])
        self.df = pd.DataFrame(data=data[0:,0:], index=data[0:,0], columns=['id', 'title', 'desc', 'asgie_title_pt'])
        self.df['text_contents'] = self.df[['title', 'desc']].apply(lambda x: " ".join(x), axis=1)
        if self.save_raw_data_to_csv:
            logging.debug("Saving raw data to CSV [%s]..." % self.save_raw_data_filename)
            self.df.to_csv('videos.csv', encoding='utf-8', sep=',', index=False)
        logging.debug("Data Loaded! Number of items: {0}".format(len(self.df)))


    # Vectorize data and reduce dimensionality
    def vectorize(self):
        """
        Vectorize training data, i.e. perform a 2-gram feature extraction and selection using a TF-IDF method 
        :return: Result is a numeric and weighted feature vector notation for each item
        """
        logging.debug("Vectorizing text contents...")
        vectorizer = TfidfVectorizer(ngram_range=(1,2), tokenizer=self.tokenizer, min_df=2)
        self.X_tfidf_matrix = vectorizer.fit_transform(self.df['text_contents'])
        self.X_text_contents = self.X_tfidf_matrix.toarray()
        self.X_text_contents = np.array(self.X_text_contents, dtype=float)

        logging.debug("Number of features in total DataFrame: {0}".format(self.X_text_contents.shape[1]))


    @staticmethod
    def tokenizer(text):
		"""
		Tokenizes sequences of text and stems the tokens.
		:param text: String to tokenize
		:return: List with stemmed tokens
		"""
		PORTUGUESE_STOP_WORDS = nltk.corpus.stopwords.words("portuguese")
		STEMMER = nltk.stem.RSLPStemmer()
		MINIMUM_WORD_LENGTH = 3
		# tokenize all email lines - separate words considering punctuation
		tokenized = nltk.tokenize.word_tokenize(text)

		# put all words in a set (unique values)
		# put only word stems
		# ignore stop words (common words with no important meaning - e.g. prepositions, pronouns)
		stems = [STEMMER.stem(word) for word in tokenized if word not in PORTUGUESE_STOP_WORDS and len(word) > MINIMUM_WORD_LENGTH]

		return stems


    def reduce_dimensionality(self, X, n_features):
        """
        Apply SVD to reduce dimension to n_features.
        :param X:
        :param n_features:
        :return:
        """
        reducer = TruncatedSVD(n_components=int(n_features))
        # Fit and transform data to n_features-dimensional space
        reducer.fit(X)
        X = reducer.transform(X)
        logging.debug("Reduced number of features to {0}".format(n_features))
        logging.debug("Percentage explained: %s\n" % reducer.explained_variance_ratio_.sum())

        return X


    def prepare_dataframe(self, X):
        """
        Prepare DataFrame for further use, e.g. finding similar items or plotting graphs.
        :param X:
        :return: Dataframe with all data and its corresponding vectorized coordinates
        """
        df_vectors = pd.DataFrame(None)
        df_vectors['title'] = self.df['title']
        df_vectors['asgie_title_pt'] = self.df['asgie_title_pt']
        df_vectors['numbers'] = range(0, len(df_vectors))
        df_vectors['coordinates'] = df_vectors['numbers'].apply(lambda index: X[index, :])
        del df_vectors['numbers']
        # Initialize dataframe by appending new columns to store the titles of the n most similar items
        for i in range(0, self.n_most_similar):
            df_vectors['most_similar_'+str(i+1)] = ""
        return df_vectors

    # Visualize data
    def visualize_data(self):
        """
        Transform the DataFrame to the 2-dimensional case and visualizes the data. The first titles are used as labels.
        :return:
        """
        logging.debug("Preparing visualization of DataFrame...")
        # Reduce dimensionality to 2 features for visualization purposes
        X_visualization = self.reduce_dimensionality(self.X, n_features=3)
        df = self.prepare_dataframe(X_visualization)
        # Set X and Y coordinate for each item
        df['X coordinate'] = df['coordinates'].apply(lambda x: x[0])
        df['Y coordinate'] = df['coordinates'].apply(lambda x: x[1])
        df['Z coordinate'] = df['coordinates'].apply(lambda x: x[2])
        # Create a list of markers, each tag has its own marker
        n_asgie_title_pt = len(self.df['asgie_title_pt'].unique())
        markers_choice_list = ['o', 's', '^', '.', 'v', '<', '>']
        markers_list = [markers_choice_list[i % 7] for i in range(n_asgie_title_pt)]
        # Create scatter plot
        sns.lmplot("X coordinate",
                   "Y coordinate",
                   hue="asgie_title_pt",
                   data=df,
                   fit_reg=False,
                   markers=markers_list,
                   scatter_kws={"s": 150})
        # Adjust borders and add title
        sns.set(font_scale=2)
        plt.title('Visualization of +TV4E Informative Videos in a 2-dimensional space')
        plt.subplots_adjust(right=0.80, top=0.90, left=0.12, bottom=0.12)
        # Show plot
        plt.show()


    # Train recommender
    def find_similar(self):
        """
        Find the n most similar items for each item in the DataFrame
        :return:
        """
        logging.debug("Calculating similarities with [{}] items...".format(self.n_most_similar))
        # Prepare DataFrame by assigning each item in the DataFrame its corresponding coordinates
        self.df_vectors = self.prepare_dataframe(self.X)
        # Calculate similarity for all TMT item and define the n most similar items
        self.calculate_similarity_scores()
        # Find the n most similar items using the similarity score dictionary
        self.find_n_most_similar()
        # Remove redundant columns
        del self.df_vectors['coordinates']

    def calculate_similarity_scores(self):
        """
        Calculate the similarity scores of all items compared to all other items.
        :return:
        """
        # Iterate over each item in DataFrame
        for index1, row1 in self.df_vectors.iterrows():
            # Initialize a dict to store the similarity scores to all other items in
            similarity_scores = {}
            # Iterate again over all items to calculate the similarity between items 1 and 2
            for index2, row2 in self.df_vectors.iterrows():
                if index1 != index2:
                    similarity_scores[index2] = self.calculate_similarity(row1['coordinates'], row2['coordinates'])
            # Save in dictionary
            self.similarity_score_dict[index1] = similarity_scores

    def find_n_most_similar(self):
        """
        Find the n most similar items with the highest similarity score for each item in the DataFrame.
        :return:
        """
        # Iterate over each item in DataFrame
        for index, row in self.df_vectors.iterrows():
            # Get the similarity scores of the current item compared to all other items
            similarity_scores = self.similarity_score_dict[index]
            # Find the highest similarity scores in the similarity_score_dict until we have found the n most similar.
            for i in range(0, self.n_most_similar):
                # Find most similar item, i.e. with highest cosine similarity. Note: if Euclidean distance, then min!
                most_similar_index = max(similarity_scores, key=similarity_scores.get)
                most_similar_score = similarity_scores[most_similar_index]
                del similarity_scores[most_similar_index]
                # Find corresponding title and set it as most similar item i in DataFrame
                # title = self.df_vectors.loc[most_similar_index]['title'].encode('utf-8')
                title_plus_score = "{} {:.2f}".format(most_similar_index, most_similar_score)
                self.df_vectors.set_value(index, 'most_similar_'+str(i+1), title_plus_score)

                self._redis.rpush(index, title_plus_score)

    def calculate_similarity(self, i1, i2):
        """
        Calculate the similarity between two items, e.g. the cosine similarity or the Euclidean distance.
        :param i1: coordinates (feature values) of item 1
        :param i2: coordinates (feature values) of item 2
        :return:
        """
        similarity = self.cosine_similarity(i1, i2) 
        similarity = "{0:.2f}".format(round(similarity, 2))
        return float(similarity)

    @staticmethod
    def cosine_similarity(x, y):
    	"""
    	Cosine similarity formula
    	"""
        def square_rooted(v):
            return round(sqrt(sum([a * a for a in v])), 3)
       
        numerator = sum(a * b for a, b in zip(x, y))
        denominator = square_rooted(x) * square_rooted(y)
        return round(numerator/float(denominator), 3)

    def save_output_to_csv(self):
        """
        Save output DataFrame to csv file
        :return:
        """
        logging.debug("Saving CSV file...")
        file_name = 'output.csv'
        try:
            self.df_vectors.to_csv(file_name, encoding='utf-8', sep=',')
        except IOError:
            logging.warning("Error while trying to save output file to %s!" % file_name)





if __name__ == "__main__":
    ContentBasedRecommender().run()
