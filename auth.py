import sqlite3
import hashlib
import streamlit as st

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Initialize SQLite database connection and create tables if they don't exist
def get_db_connection():
    conn = sqlite3.connect('users.db')
    return conn

def create_tables():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages (username TEXT, role TEXT, content TEXT)''')
    conn.commit()
    conn.close()

create_tables()

# User authentication functions
def user_exists(username):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    exists = c.fetchone() is not None
    conn.close()
    return exists

def register_user(username, password):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
    conn.commit()
    conn.close()

def authenticate_user(username, password):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hash_password(password)))
    authenticated = c.fetchone() is not None
    conn.close()
    return authenticated

def login_ui():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if 'show_registration' not in st.session_state:
        st.session_state.show_registration = False

    if not st.session_state.authenticated:
        st.subheader("Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.messages = [{"role": "assistant", "content": "Hi I'm your AI Mental Health Advisor"}]
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("SELECT role, content FROM messages WHERE username=?", (username,))
                for row in c.fetchall():
                    st.session_state.messages.append({"role": row[0], "content": row[1]})
                conn.close()
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")

        if st.button("New user? Register"):
            st.session_state.show_registration = True

        if st.session_state.show_registration:
            st.subheader("Register")
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            if st.button("Register"):
                if user_exists(new_username):
                    st.error("Username already exists")
                else:
                    register_user(new_username, new_password)
                    st.success("Registration successful")
    return st.session_state.authenticated
