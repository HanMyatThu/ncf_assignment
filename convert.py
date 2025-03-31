import pandas as pd


# convert all dat to csv files
# ratings
ratings = pd.read_csv('./dataset/ratings.dat', sep='::', engine='python',
                      names=['userId', 'movieId', 'rating', 'timestamp'])

ratings.to_csv('./dataset/ratings.csv', index=False)

# users
users = pd.read_csv('./dataset/users.dat', sep='::', engine='python',
                    names=['userId', 'gender', 'age', 'occupation', 'zipCode'])

users.to_csv('./dataset/users.csv', index=False)

# movies
movies = pd.read_csv('./dataset/movies.dat', sep='::', engine='python',
                     names=['movieId', 'title', 'genres'], encoding='ISO-8859-1')

movies.to_csv('./dataset/movies.csv', index=False)

print("All Dat files are converted to csv")