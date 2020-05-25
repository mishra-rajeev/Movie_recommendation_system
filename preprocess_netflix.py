import pandas as pd
import numpy as np

data = pd.read_csv('netflix_titles.csv')

# keeping the columns that are useful in recommndation system
data = data.loc[:,['title','director','cast','country','listed_in']]

# replacing null values in the all columns with string 'unknown'
data['director'] = data['director'].replace(np.nan, 'unknown')
data['cast'] = data['cast'].replace(np.nan, 'unknown')
data['country'] = data['country'].replace(np.nan, 'unknown')

# In the ‘cast’ column, replacing the ‘,’ with whitespace,
# so the casts would be considered different strings.
data['cast'] = data['cast'].str.replace(',', ' ')

data['country'] = data['country'].str.replace(',', ' ')
data['listed_in'] = data['listed_in'].str.replace(',', ' ')

# Now converting the ‘movie_title’ columns values to lowercase for searching simplicity.
data['title'] = data['title'].str.lower()

# making the new column containing combination of all the features
data['comb'] = data['cast'] + ' '+ data['country'] + ' '+ data['listed_in'] + ' '+ data['director']

# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# creating a count matrix
cv = CountVectorizer()
count_matrix = cv.fit_transform(data['comb'])

# creating a similarity score matrix
sim = cosine_similarity(count_matrix)

# saving the similarity score matrix in a file for later use
np.save('similarity_matrix_netflix', sim)

# saving dataframe to csv for later use in main file
data.to_csv('data_netflix.csv', index=False)





