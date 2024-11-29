import random
from models import Session, Rating

def generate_ratings_data(num_users, num_books, num_entries):
    ratings_data = []
    seen_pairs = set()

    while len(ratings_data) < num_entries:
        user_id = random.randint(1, num_users)
        book_id = random.randint(1, num_books)
        if (user_id, book_id) not in seen_pairs:
            seen_pairs.add((user_id, book_id))
            rating = round(random.uniform(1.0, 5.0), 1)
            ratings_data.append({'user_id': user_id, 'book_id': book_id, 'rating': rating})
    
    return ratings_data

def populate_ratings():
    session = Session()

    # Generate 5000 sample ratings data entries
    num_users = 51
    num_books = 7928
    num_entries = 5000

    ratings_data = generate_ratings_data(num_users, num_books, num_entries)

    # Add ratings to the database
    for data in ratings_data:
        rating = Rating(user_id=data['user_id'], book_id=data['book_id'], rating=data['rating'])
        session.add(rating)

    session.commit()
    session.close()

if __name__ == '__main__':
    populate_ratings()
