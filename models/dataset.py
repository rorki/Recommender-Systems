from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

iid = 'iid__'
uid = 'uid__'
rate = 'rate__'


class DataSet:

    def __init__(self, df_ratings, rating_cols, df_reviews, review_cols,
                 vectorize_reviews=True, empty_element='', noise_reviews = False,
                 matrix_noise = 0.3):
        """
        DataSet manages an internal uui and iid and provides universal access to data from different data sets.
        Makes dataset changes in place!!!

        :param df_ratings: pandas data frame with ratings
        :param rating_cols: column names in order < user_id, item_id, rating>
        :param df_reviews: pandas data frame with reviews
        :param review_cols: column names in order <item_id, review_text>
        """
        # adjust DataFrame
        df_ratings.rename(columns={rating_cols[2]: rate}, inplace=True)

        # convert user and item ids to inner ids
        self.uid_map = {v: k for k, v in enumerate(df_ratings[rating_cols[0]].unique())}
        self.iid_map = {v: k for k, v in enumerate(df_ratings[rating_cols[1]].unique())}

        df_ratings[iid] = df_ratings[rating_cols[1]].map(self.iid_map)
        df_ratings[uid] = df_ratings[rating_cols[0]].map(self.uid_map)

        # split to train, validation and test set
        trainset, testset = train_test_split(df_ratings, test_size=0.3, random_state=42)
        trainset, valset = train_test_split(trainset, test_size=0.2, random_state=42)

        # drop rows with items\user not presented in train set
        self.trainset = trainset
        testset = testset[testset[uid].isin(trainset[uid])]
        self.testset = testset[testset[iid].isin(trainset[iid])]
        valset = valset[valset[uid].isin(trainset[uid])]
        self.valset = valset[valset[iid].isin(trainset[iid])]

        # fill in empty reviews for missed items
        empty_reviews = []
        diff = np.setdiff1d(df_ratings[rating_cols[1]].unique(), df_reviews[review_cols[0]].unique())
        for id_ in diff:
            df_reviews.loc[df_reviews.index.argmax() + 1] = [id_, empty_element]
            empty_reviews.append(id_)
        print('Filled in %s empty reviews: %s...' % (len(empty_reviews), empty_reviews[0:5]))

        df_reviews[iid] = df_reviews[review_cols[0]].map(self.iid_map)
        df_reviews.dropna(subset=[iid], inplace=True)
        df_reviews.sort_values(by=[iid], inplace=True)

        if vectorize_reviews:
            vectorizer = TfidfVectorizer(max_features=10000)
            self.review_matrix = np.array(vectorizer.fit_transform(df_reviews[review_cols[1]].values).todense())
        else:
            self.review_matrix = np.array(df_reviews[review_cols[1]].values.tolist(), dtype=int)

        if noise_reviews:
            self.noised_review_matrix = self.add_noise(self.review_matrix, matrix_noise)

    def get_train_rating_matrix(self):
        return self.trainset[[uid, iid, rate]].values

    def get_valid_rating_matrix(self):
        return self.valset[[uid, iid, rate]].values

    def get_test_rating_matrix(self):
        return self.testset[[uid, iid, rate]].values

    def train_user_num(self):
        return len(self.uid_map)

    def train_item_num(self):
        return len(self.iid_map)

    @staticmethod
    def mask(x, corruption_level):
        mask = np.random.binomial(1, 1 - corruption_level, x.shape)
        return np.multiply(x, mask)

    @staticmethod
    def add_noise(x, corruption_level):
        print('Noising of reviews')
        return np.array([DataSet.mask(x=i, corruption_level=corruption_level) for i in x])


