from models import Session, User

def inspect_users():
    session_db = Session()
    users = session_db.query(User).all()
    for user in users:
        print(f"User ID: {user.id}, Username: {user.username}, Password Hash: {user.password_hash}")
    session_db.close()

if __name__ == '__main__':
    inspect_users()