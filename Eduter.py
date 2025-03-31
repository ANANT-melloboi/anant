import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
import pickle
import sqlite3
import hashlib
import os

def create_users_table():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT
                 )''')
    conn.commit()
    conn.close()

MODEL_FILENAME = 'performance_model.pkl'

def train_and_save_model():
    data = np.array([
        [40, 70, 9.0], [42, 68, 8.9], [45, 75, 9.7], [38, 65, 8.5], [50, 80, 9.8],
        [30, 50, 6.5], [35, 60, 7.5], [48, 78, 9.8], [44, 72, 9.4], [46, 74, 9.7]
    ])
    X = data[:, :-1]
    y = data[:, -1]

    model = LinearRegression()
    model.fit(X, y)

    with open(MODEL_FILENAME, 'wb') as file:
        pickle.dump(model, file)
    return model

def load_model():
    if not os.path.exists(MODEL_FILENAME):
        model = train_and_save_model()
    else:
        try:
            with open(MODEL_FILENAME, 'rb') as file:
                model = pickle.load(file)
        except Exception:
            model = train_and_save_model()
    return model

def analyze_performance(model, average_internals, average_externals):
    input_data = np.array([[average_internals, average_externals]])
    predicted_cgpa = model.predict(input_data)
    return max(0, min(predicted_cgpa[0], 10))

def add_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        st.success("✅ Account created successfully! You can now log in.")
    except sqlite3.IntegrityError:
        st.error(f"⚠ Username '{username}' already exists. Please choose a different one.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))
    user = c.fetchone()
    conn.close()
    return user is not None

def login(username, password):
    if not username or not password:
         st.warning("Please enter both username and password.")
         return
    if authenticate_user(username, password):
        st.session_state["logged_in"] = True
        st.session_state["username"] = username
        st.rerun()
    else:
        st.error("❌ Invalid username or password")

def sign_up(username, password):
    if not username or not password:
         st.warning("Please enter both username and password.")
         return
    add_user(username, password)

create_users_table()
model = load_model()

st.title("📊 EduCom - Your Future is Worth Blossoming")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        st.subheader("Login")
        login_username = st.text_input("Username:", key="login_user")
        login_password = st.text_input("Password:", type="password", key="login_pass")
        if st.button("Login", key="login_button"):
            login(login_username, login_password)

    with signup_tab:
        st.subheader("Create Account")
        signup_username = st.text_input("Choose a Username:", key="signup_user")
        signup_password = st.text_input("Choose a Password:", type="password", key="signup_pass")
        if st.button("Sign Up", key="signup_button"):
            sign_up(signup_username, signup_password)

else:
    st.sidebar.success(f"Logged in as: {st.session_state.get('username', 'User')}")
    if st.sidebar.button("Sign Out"):
        for key in ["logged_in", "username"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.header("🎓 Performance Analysis")

    user_level = st.selectbox("Select your level:", ["School", "College"])
    term_label = "Semester" if user_level == "College" else "Term"
    max_terms = 8 if user_level == "College" else 4

    student_name = st.text_input("Enter Student Name:")

    num_terms = st.slider(f"Select Number of {term_label}s:", 1, max_terms, 2)

    all_avg_internals = []
    all_avg_externals = []

    cols = st.columns(num_terms)

    for term_idx in range(num_terms):
        with cols[term_idx]:
            st.subheader(f"📌 {term_label} {term_idx + 1}")

            total_marks_internal = st.number_input(
                f"Max Internal Marks per Subject:",
                min_value=1, value=50, step=1, key=f"total_marks_int_{term_idx}"
            )
            num_internals = st.number_input(
                f"Number of Internals:",
                min_value=1, value = 2, step=1, key=f"num_internals_{term_idx}"
            )

            internal_marks_list = []
            for i in range(int(num_internals)):
                mark = st.number_input(
                    f"Internal {i+1} Marks:",
                    min_value=0.0,
                    max_value=float(total_marks_internal),
                    step=0.5,
                    key=f"internal_marks_{term_idx}_{i}"
                )
                internal_marks_list.append(mark)

            avg_internals = np.mean(internal_marks_list) if internal_marks_list else 0
            all_avg_internals.append(avg_internals)

            total_marks_external = st.number_input(
                f"Max External Marks:",
                min_value=1, value=100, step=1, key=f"total_marks_ext_{term_idx}"
            )
            avg_externals = st.number_input(
                f"Average External Marks:",
                min_value=0.0,
                max_value=float(total_marks_external),
                step=0.5,
                key=f"external_marks_{term_idx}"
            )
            all_avg_externals.append(avg_externals)


    if st.button("Analyze Performance"):
        if not student_name:
            st.warning("Please enter the student's name.")
        elif len(all_avg_internals) != num_terms or len(all_avg_externals) != num_terms:
             st.warning(f"Please ensure you have entered marks for all {num_terms} {term_label}s.")
        elif model is None:
            st.error("Model could not be loaded. Please check server logs.")
        else:
            predicted_scores = [
                analyze_performance(model, avg_int, avg_ext)
                for avg_int, avg_ext in zip(all_avg_internals, all_avg_externals)
            ]

            st.write(f"### 📈 Predicted Performance for {student_name}")
            results_cols = st.columns(num_terms)
            for i, score in enumerate(predicted_scores):
                 with results_cols[i]:
                     st.metric(label=f"{term_label} {i+1}", value=f"{score:.2f}/10")

            st.write("### 📊 Performance Trend")
            term_numbers = range(1, num_terms + 1)

            fig_line, ax_line = plt.subplots(figsize=(10, 5))
            ax_line.plot(term_numbers, predicted_scores, marker='o', linestyle='-', color='dodgerblue', linewidth=2, markersize=8, label="Predicted Score")
            ax_line.set_xticks(term_numbers)
            ax_line.set_xlabel(term_label)
            ax_line.set_ylabel('Predicted Performance Score (out of 10)')
            ax_line.set_ylim(0, 10.5)
            ax_line.set_title(f'Predicted Performance Trend for {student_name}')
            ax_line.legend()
            ax_line.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig_line)