# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 18:24:35 2017

@author: AayKayZee

"""
#importing necessary libraries
import pandas as pd
import numpy as np

#importing the data files into dataframes
users_df = pd.read_csv('u.user', sep = '|', names = ['UserID','age','gender','occupation','zip code'], encoding = 'latin-1')
movies_df = pd.read_csv('u.item', sep = '|', names = ['MovieID','movie title','release date','video release date','IMDb URL','unknown','Action','Adventure','Animation','Childrens','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance ','Sci-Fi','Thriller','War' ,'Western'],encoding = 'latin-1')
ratings_df = pd.read_csv('u.data', sep = '\t', names = ['UserID','MovieID','Rating','Timestamp'], encoding = 'latin-1' )

#Checking the dataframes
movies_df.head()
ratings_df.head()

#Changing the dataframe into one row per user and one column per movie
Ratings = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
Ratings.head()

#Noramlizing the data and convert into a numpy array
R = Ratings.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
R_norm = R - user_ratings_mean.reshape(-1, 1)

#Importing the Single Value Decompositon svds
from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_norm, k = 50)
"""
Vt represents how relevant is each feature to the movie.
U represents how much the user like each feature
sigma represents the weights of these features.
k represents the no. of latency factors to be considered
"""
#Converting to diagonal matrix
sigma = np.diag(sigma)

#Making Recommendations
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = Ratings.columns)
preds_df.head()

#function for recommendation engine 
def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 # UserID starts at 1, not 0
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False) # UserID starts at 1
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.UserID == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'MovieID', right_on = 'MovieID').
                     sort_values(['Rating'], ascending=False)
                 )

    print ('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print ('Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations))
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'MovieID',
               right_on = 'MovieID').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations


#The function is called on the user with the necessary arguments
already_rated, predictions = recommend_movies(preds_df, 83, movies_df, ratings_df, 10)

already_rated.head(10)#The movie already rated by the user 
predictions#The predicted movies
