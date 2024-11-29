import pandas as pd
import pickle
from models import Session, Book

# Load the pre-trained model and user-book matrix
with open('model.pkl', 'rb') as f:
    model, user_book_matrix = pickle.load(f)

def get_recommendations(book_title, books_df, n_recommendations=10):
    # Find the book ID for the given title
    book_id = books_df[books_df['title'] == book_title].index[0]

    # Get the user ratings for the specified book
    book_ratings = user_book_matrix.loc[:, book_id].values.reshape(1, -1)

    # Set the number of neighbors
    n_neighbors = min(n_recommendations + 1, len(user_book_matrix))
    if n_neighbors > 1:
        distances, indices = model.kneighbors(book_ratings, n_neighbors=n_neighbors)

        # Get the book indices of the recommended books
        recommended_book_ids = indices.flatten()[1:]

        # Return the recommended books
        return books_df.iloc[recommended_book_ids]
    else:
        return pd.DataFrame(columns=books_df.columns)
