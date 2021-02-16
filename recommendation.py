#!/usr/bin/env python3.6

import pandas as pd
import numpy as np
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import BaselineOnly
from surprise.accuracy import rmse
from surprise import accuracy

from collections import defaultdict

from sklearn.model_selection import train_test_split


df = pd.read_csv('Food_data.csv')
print(df.head())
print('\n')

users = df['UserID'].value_counts()>=50
users = users[users].index.tolist()

items = df['Food_item_ID'].value_counts()>=50
items = items[items].index.tolist()

ratings_w_titles = df[(df['Food_item_ID'].isin(items)) & (df['UserID'].isin(users))]
ratings_w_titles.reset_index(drop= True, inplace= True)


id_rest = input("Please enter a restaurant id:")
user = int(input("Please enter a user id:"))


def popularity_recommendation():

    class Popularity_Recommender():
        def __init__(self):
            self.df = None
            self.user_id = None
            self.product_id = None
            self.popularity_recommendations = None

        #Create the popularity based recommender system model
        def create(self, df, user_id, product_id):
            self.df = df
            self.user_id = user_id
            self.product_id = product_id

            #Get a count of user_ids for each unique product as recommendation score
            train_data_grouped = df.groupby([self.product_id]).agg({self.user_id: 'count'}).reset_index()
            train_data_grouped.rename(columns = {'UserID': 'Score'},inplace=True)

            #Sort the products based upon recommendation score
            train_data_sort = train_data_grouped.sort_values(['Score', self.product_id], ascending = [0,1])

            #Generate a recommendation rank based upon score
            train_data_sort['Rank'] = train_data_sort['Score'].rank(ascending=0, method='first')

            #Get the top 10 recommendations
            self.popularity_recommendations = train_data_sort.head(10)

        #Use the popularity based recommender system model to
        #make recommendations
        def recommend(self, user_id):    
            user_recommendations = self.popularity_recommendations

            #Add user_id column for which the recommendations are being generated
#             user_recommendations['UserID'] = user_id

            #Bring user_id column to the front
            cols = user_recommendations.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            user_recommendations = user_recommendations[cols]

            return (user_recommendations)

    pop_recommend = Popularity_Recommender()
    pop_recommend.create(df, 'UserID', 'Food_item_ID')

    print(pop_recommend.recommend(896))

# Collaborative Reccomendation

def collaborative_recommendation(ratings_w_titles, minimum, maximum, userid):

    reader = Reader(rating_scale=(minimum, maximum))
    data = Dataset.load_from_df(ratings_w_titles, reader)

    from surprise.model_selection import train_test_split
    train_df, test_df = train_test_split(data, test_size=.95)

    # Use the new parameters with the train test data
    bsl_options = {'method': 'als',
               'n_epochs': 10,
               'reg_u': 12,
               'reg_i': 5
               }
    algo = BaselineOnly(bsl_options= bsl_options)
    algo.fit(train_df)
    test_pred = algo.test(test_df)

    def get_top_n(predictions, n=5):
        '''Return the top-N recommendation for each user from a set of predictions.
        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.
        Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        '''

        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, _, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n

    top_n = get_top_n(test_pred, n=10)
    #Print the recommended food item id
    
    for i in range(len(top_n[userid])):
        print(top_n[userid][i][0])
    
try:
    groups = ratings_w_titles.groupby(['Restaurant_ID'])
    ratings_w_titles = groups.get_group(int(id_rest))
    if user in ratings_w_titles['UserID'].unique():
        minimum = ratings_w_titles['Ratings'].min()
        maximum = ratings_w_titles['Ratings'].max()
        ratings_w_titles.drop(['Ratings'], axis= 1, inplace = True)
        collaborative_recommendation(ratings_w_titles, minimum, maximum, user)
    else:
        popularity_recommendation()
# print(ratings_w_titles.head())
except KeyError:
    print('Please enter a valid resturant id')