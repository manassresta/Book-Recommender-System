import pandas as pd
from models import Session, Book

def load_data():
    session = Session()

    # Load dataset
    df = pd.read_csv('data/books.csv')

    # Print the column names to verify
    print("Columns in the CSV file:", df.columns)

    # Strip any leading/trailing spaces in the column names
    df.columns = df.columns.str.strip()

    # Verify that the expected columns are present
    required_columns = {'Title', 'Author', 'Main Genre', 'Rating', 'URLs'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV file is missing one or more required columns: {required_columns}")

    # Add books to the database
    for _, row in df.iterrows():
        book = Book(
            title=row['Title'],
            author=row['Author'],
            main_genre=row['Main Genre'],
            rating=row['Rating'],
            url=row['URLs']
        )
        session.add(book)

    session.commit()
    session.close()

if __name__ == '__main__':
    load_data()
