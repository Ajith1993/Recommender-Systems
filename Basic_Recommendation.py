# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 15:03:34 2017

@author: AayKayZee
"""
"""
This program recommend the movies according to the Average ratings and No. of votes.The top 5 movies are recommended to the User.

This is a generalized recommendation.It does not change for each user
"""
#importing  relevant librarires 
import pandas as pd

#importing the data files into dataframes
user = pd.read_csv('u.user',sep = '|', names = ['user id','age','gender','occupation','zip code'], encoding = 'latin-1')
item = pd.read_csv('u.item', sep = '|', names = ['movie id','movie title','release date','video release date','IMDb URL','unknown','Action','Adventure','Animation','Childrens','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance ','Sci-Fi','Thriller','War' ,'Western'],encoding = 'latin-1')
data = pd.read_csv('u.data', sep = '\t', names = ['user id','movie id','rating','timestamp'], encoding = 'latin-1' )

#printing the details of these dataframes() 
print('\n',user.info())
print('\n',item.info())
print('\n',data.info())

#Data Manipulation

#Merging three dataframes inot one 
df= pd.merge(pd.merge(item, data), user)

#Finding Mean ratings and total no.of ratings

#To find the no. of elements in the dataframe
tot_rt = df.groupby('movie title').size()

#To find the mean of ratings arranged by Movie Title
rt = (df.groupby('movie title')['movie title', 'rating'])
mean_rt = rt.mean()

#Changing the panda Series to Dataframe format
tot_rt = pd.DataFrame({'movie title': tot_rt.index,'total ratings':tot_rt.values})
mean_rt['movie title'] = mean_rt.index

#Merge the dataframes to create the ranking

#Sorting the movies according to the no. of votes
votes_rank = pd.merge(mean_rt, tot_rt).sort_values(by = 'total ratings',ascending = False)
print('Ranking : No. of Votes\n')
print(votes_rank.head())
print('\n')

#Sorting the movies according to the average rating
avg_rank = votes_rank[:100].sort_values(by = 'rating', ascending = False)
print('Ranking :Average Rating\n')
print(avg_rank.head())

