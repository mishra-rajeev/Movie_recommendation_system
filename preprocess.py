import pandas as pd
import numpy as np

data = pd.read_csv('movie_metadata.csv')

# Extracting columns that are useful in recommndation system
data = data.loc[:, ['movie_title', 'director_name', 'language',
                    'country', 'genres', 'actor_1_name', 'actor_2_name', 'actor_3_name']]

# replacing null values in the all columns with string 'unknown'
data['director_name'] = data['director_name'].replace(np.nan, 'unknown')
data['language'] = data['language'].replace(np.nan, 'unknown')
data['country'] = data['country'].replace(np.nan, 'unknown')
data['actor_1_name'] = data['actor_1_name'].replace(np.nan, 'unknown')
data['actor_2_name'] = data['actor_2_name'].replace(np.nan, 'unknown')
data['actor_3_name'] = data['actor_3_name'].replace(np.nan, 'unknown')

# In the ‘genres’ column, replacing the ‘,’ with whitespace,
# so the genres would be considered different strings.
data['genres'] = data['genres'].str.replace('|', ' ')

# Removing last char in movie title
data['movie_title'] = data['movie_title'].str[:-1]

# Now converting the ‘movie_title’ columns values to lowercase for searching simplicity.
data['movie_title'] = data['movie_title'].str.lower()

# making the new column containing combination of all the features
data['comb'] = data['director_name'] + ' ' + data['language'] + ' ' + data['country'] + ' ' + \
    data['genres'] + ' ' + data['actor_1_name'] + ' ' + \
    data['actor_2_name'] + ' ' + data['actor_3_name']

# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# creating a count matrix
cv = CountVectorizer()
count_matrix = cv.fit_transform(data['comb'])

# creating a similarity score matrix
sim = cosine_similarity(count_matrix)

# saving the similarity score matrix in a file for later use
np.save('similarity_matrix', sim)

# saving dataframe to csv for later use in main file
data.to_csv('data.csv', index=False)

