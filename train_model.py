import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pickle
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import warnings

from models import Rating

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def normalize_ratings(ratings_df):
    rating_mean = ratings_df['rating'].mean()
    rating_std = ratings_df['rating'].std()
    ratings_df['rating_norm'] = (ratings_df['rating'] - rating_mean) / rating_std
    return ratings_df, rating_mean, rating_std

def split_data(ratings_df, test_size=0.2):
    train_df, test_df = train_test_split(ratings_df, test_size=test_size, random_state=42)
    return train_df, test_df

def calculate_metrics(y_true, predictions, threshold=3.0):
    y_true_binary = (np.array(y_true) >= threshold).astype(int)
    y_pred_binary = (np.array(predictions) >= threshold).astype(int)
    
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=1)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=1)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=1)
    
    return accuracy, precision, recall, f1

def train_model():
    # Database setup (update the connection string accordingly)
    engine = create_engine('sqlite:///bookrec.db')
    Session = sessionmaker(bind=engine)
    session = Session()

    # Fetch data from the database
    try:
        ratings = session.query(Rating).all()
    except Exception as e:
        print(f"Error querying the Rating table: {e}")
        session.close()
        return

    # Convert to DataFrame
    ratings_list = [{'id': index, 'user_id': r.user_id, 'book_id': r.book_id, 'rating': r.rating} for index, r in enumerate(ratings)]

    if not ratings_list:
        print("No ratings found in the database.")
        session.close()
        return

    ratings_df = pd.DataFrame(ratings_list)

    # Normalize ratings
    ratings_df, rating_mean, rating_std = normalize_ratings(ratings_df)

    # Remove duplicates by keeping the latest rating
    ratings_df = ratings_df.sort_values('id').drop_duplicates(subset=['user_id', 'book_id'], keep='last')

    # Split data into training and test sets
    train_df, test_df = split_data(ratings_df)

    # Pivot the table to create user-item matrices for training and test sets
    user_book_matrix_train = train_df.pivot(index='user_id', columns='book_id', values='rating_norm').fillna(0)
    user_book_matrix_test = test_df.pivot(index='user_id', columns='book_id', values='rating_norm').fillna(0)

    # Ensure column names are strings
    user_book_matrix_train.columns = user_book_matrix_train.columns.astype(str)
    user_book_matrix_test.columns = user_book_matrix_test.columns.astype(str)

    # Calculate item similarity matrix
    similarity_matrix = user_book_matrix_train.corr(method='pearson').fillna(0)

    # Sample of similarity matrix for debugging
    print("Sample of item similarity matrix:")
    print(similarity_matrix.iloc[:5, :5])

    # Make predictions for each user-item pair in the test set
    predictions = []
    y_true = []
    
    for index, row in test_df.iterrows():
        user_id = row['user_id']
        book_id = str(row['book_id'])  # Ensure book_id is a string to match the column names

        if user_id in user_book_matrix_train.index and book_id in similarity_matrix.columns:
            similar_books = similarity_matrix[book_id].sort_values(ascending=False)
            similar_books = similar_books[similar_books.index != book_id]  # Exclude the book itself

            # Only consider top 5 similar books
            top_similar_books = similar_books.head(5)
            top_similar_books_ratings = user_book_matrix_train.loc[user_id, top_similar_books.index]

            if not top_similar_books_ratings.empty and top_similar_books_ratings.sum() != 0:
                predicted_rating_norm = np.dot(top_similar_books, top_similar_books_ratings) / top_similar_books_ratings.sum()
                predicted_rating = (predicted_rating_norm * rating_std) + rating_mean
            else:
                predicted_rating = rating_mean  # Default to mean rating if no similar books or ratings available
        else:
            predicted_rating = rating_mean  # Default to mean rating if user or book not in training set

        predictions.append(predicted_rating)
        y_true.append(row['rating'])

    # Evaluate metrics
    accuracy, precision, recall, f1 = calculate_metrics(y_true, predictions)

    # Print the evaluation metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Save the model and user-book matrix to a file
    with open('model.pkl', 'wb') as f:
        pickle.dump((similarity_matrix, user_book_matrix_train, rating_mean, rating_std), f)

    session.close()

if __name__ == '__main__':
    train_model()
