import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session as flask_session
from sqlalchemy import func
from models import Session, User, Book, Rating
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load the trained model and user-book matrix
with open('model.pkl', 'rb') as f:
    similarity_matrix, user_book_matrix, rating_mean, rating_std = pickle.load(f)

@app.route('/')
def home():
    session_db = Session()
    top_books = session_db.query(Book).order_by(Book.rating.desc()).limit(50).all()
    
    avg_ratings_dict = {}
    for book in top_books:
        ratings = session_db.query(Rating).filter_by(book_id=book.id).all()
        if ratings:
            avg_rating = sum(r.rating for r in ratings) / len(ratings)
        else:
            avg_rating = 0.0
        avg_ratings_dict[book.id] = round(avg_rating, 2)
        print(f"Book ID: {book.id}, Average Rating: {avg_ratings_dict[book.id]}")  # Debug print
    
    session_db.close()
    return render_template('home.html', books=top_books, avg_ratings=avg_ratings_dict)



@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        age = request.form['age']
        preferred_genre = request.form['preferred_genre']
        session_db = Session()

        if session_db.query(User).filter_by(username=username).first():
            session_db.close()
            return 'Username already exists!'
        
        new_user = User(username=username, age=age, preferred_genre=preferred_genre)
        new_user.set_password(password)
        session_db.add(new_user)
        session_db.commit()
        session_db.close()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        session_db = Session()
        user = session_db.query(User).filter_by(username=username).first()
        if user is None or not user.check_password(password):
            session_db.close()
            return 'Invalid username or password'
        flask_session['user_id'] = user.id
        session_db.close()
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    flask_session.pop('user_id', None)
    return redirect(url_for('home'))
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def find_similar_users(user_id, user_book_matrix, session_db, n_neighbors=5):
    user_id_str = str(user_id)

    # Ensure user_id is in the user_book_matrix
    if user_id_str not in user_book_matrix.index:
        return []

    # Compute cosine similarity between users
    user_similarity = cosine_similarity(user_book_matrix.values)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_book_matrix.index, columns=user_book_matrix.index)

    # Get the similarity scores for the current user
    user_similarity_series = user_similarity_df.loc[user_id_str]

    # Sort users by similarity score, exclude the current user
    similar_users = user_similarity_series.sort_values(ascending=False).index[1:n_neighbors + 1]

    return similar_users

@app.route('/recommend', methods=['GET'])
def recommend_books():
    if 'user_id' not in flask_session:
        return redirect(url_for('login'))

    user_id = flask_session['user_id']
    session_db = Session()

    try:
        # Fetch user ratings
        user_ratings = session_db.query(Rating).filter_by(user_id=user_id).all()
        if not user_ratings:
            session_db.close()
            return render_template('no_recommendations.html')

        # Identify the genres of books the user has rated highly
        rated_genres = {}
        for rating in user_ratings:
            book = session_db.query(Book).filter_by(id=rating.book_id).first()
            if book:
                genre = book.main_genre
                if genre not in rated_genres:
                    rated_genres[genre] = []
                rated_genres[genre].append(rating.rating)

        # Calculate the average rating for each genre
        avg_genre_ratings = {genre: sum(ratings)/len(ratings) for genre, ratings in rated_genres.items()}

        # Identify the user's preferred genre
        preferred_genre = max(avg_genre_ratings, key=avg_genre_ratings.get)

        # Get books from the preferred genre
        recommended_books = session_db.query(Book).filter_by(main_genre=preferred_genre).order_by(Book.rating.desc()).limit(10).all()

        # Add collaborative filtering recommendations (optional)
        if len(recommended_books) < 10:
            # (Existing collaborative filtering logic here)
            similar_users = find_similar_users(user_id, user_book_matrix, session_db)
            for similar_user in similar_users:
                similar_user_ratings = session_db.query(Rating).filter_by(user_id=int(similar_user)).all()
                for rating in similar_user_ratings:
                    book = session_db.query(Book).filter_by(id=rating.book_id).first()
                    if book and book.main_genre == preferred_genre and book.id not in [b.id for b in recommended_books]:
                        recommended_books.append(book)
                        if len(recommended_books) >= 10:
                            break
                if len(recommended_books) >= 10:
                    break

        # Calculate average ratings for recommended books
        avg_ratings_dict = {}
        for book in recommended_books:
            ratings = session_db.query(Rating).filter_by(book_id=book.id).all()
            avg_rating = sum(r.rating for r in ratings) / len(ratings) if ratings else 0.0
            avg_ratings_dict[book.id] = round(avg_rating, 2)

    except Exception as e:
        print(f"Error occurred: {e}")
        recommended_books = []
        avg_ratings_dict = {}

    finally:
        session_db.close()

    return render_template('recommendations.html', books=recommended_books, avg_ratings=avg_ratings_dict)


@app.route('/rate', methods=['GET', 'POST'])
def rate_book():
    if 'user_id' not in flask_session:
        return redirect(url_for('login'))

    session_db = Session()
    if request.method == 'POST':
        book_id = request.form['book_id']
        rating = request.form['rating']
        user_id = flask_session['user_id']

        # Convert rating to float
        try:
            rating = float(rating)
        except ValueError:
            return "Invalid rating value. Please enter a numeric value."

        # Ensure user_id and book_id are integers
        try:
            user_id = int(user_id)
            book_id = int(book_id)
        except ValueError:
            return "Invalid user or book ID."

        existing_rating = session_db.query(Rating).filter_by(user_id=user_id, book_id=book_id).first()
        if existing_rating:
            existing_rating.rating = rating
        else:
            new_rating = Rating(user_id=user_id, book_id=book_id, rating=rating)
            session_db.add(new_rating)
        
        session_db.commit()

        # Update the user_book_matrix
        user_id_str = str(user_id)
        book_id_str = str(book_id)
        if book_id_str not in user_book_matrix.columns:
            user_book_matrix[book_id_str] = 0.0
        if user_id_str not in user_book_matrix.index:
            user_book_matrix.loc[user_id_str] = 0.0
        user_book_matrix.at[user_id_str, book_id_str] = rating

        # Re-train the model with the updated matrix
        user_book_matrix_combined = user_book_matrix.fillna(0)

        # Similar users based on cosine similarity
        user_similarity = user_book_matrix_combined.dot(user_book_matrix_combined.loc[user_id_str])
        user_similarity = user_similarity.sort_values(ascending=False)

        # Save the updated model and user-book matrix
        with open('model.pkl', 'wb') as f:
            pickle.dump((similarity_matrix, user_book_matrix, rating_mean, rating_std), f)

        session_db.close()
        return redirect(url_for('rating_confirmation'))

    books = session_db.query(Book).all()
    session_db.close()
    return render_template('rate.html', books=books)


@app.route('/rating_confirmation')
def rating_confirmation():
    return render_template('rating_confirmation.html')

@app.route('/account')
def account():
    if 'user_id' not in flask_session:
        return redirect(url_for('login'))

    user_id = flask_session['user_id']
    session_db = Session()
    user_ratings = session_db.query(Rating).filter_by(user_id=user_id).all()

    ratings = []
    for rating in user_ratings:
        book = session_db.query(Book).filter_by(id=rating.book_id).first()
        ratings.append({'book': book, 'rating': rating.rating})

    session_db.close()
    return render_template('account.html', ratings=ratings)


@app.route('/preference_recommendations', methods=['GET'])
def preference_recommendations():
    if 'user_id' not in flask_session:
        return redirect(url_for('login'))

    user_id = flask_session['user_id']
    session_db = Session()
    user = session_db.query(User).filter_by(id=user_id).first()

    if user is None:
        session_db.close()
        return 'User not found'
    
    age = user.age
    preferred_genre = user.preferred_genre

    # Find other users with similar age and preferred genre
    similar_users = session_db.query(User).filter(User.age.between(age - 5, age + 5),
                                                  User.preferred_genre == preferred_genre).all()

    recommended_books = []
    similar_books_ids = set()
    
    if similar_users:
        for similar_user in similar_users:
            user_ratings = session_db.query(Rating).filter_by(user_id=similar_user.id).all()
            for rating in user_ratings:
                book = session_db.query(Book).filter_by(id=rating.book_id).first()
                if book and book.main_genre == preferred_genre and book.id not in similar_books_ids:
                    recommended_books.append(book)
                    similar_books_ids.add(book.id)
    
    # Fetch all books in the preferred genre
    genre_books = session_db.query(Book).filter_by(main_genre=preferred_genre).all()
    for book in genre_books:
        if book.id not in similar_books_ids:
            recommended_books.append(book)

    # Calculate the average rating for each recommended book
    books_with_ratings = []
    for book in recommended_books:
        avg_rating = session_db.query(func.avg(Rating.rating)).filter(Rating.book_id == book.id).scalar()
        avg_rating = round(avg_rating, 2) if avg_rating else 0.0
        book.avg_rating = avg_rating
        books_with_ratings.append(book)

    session_db.close()
    return render_template('preference_recommendations.html', books=books_with_ratings)




if __name__ == '__main__':
    app.run(debug=True)