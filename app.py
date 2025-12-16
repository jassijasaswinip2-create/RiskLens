from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import seaborn as sns
import matplotlib

# IMPORTANT: non-GUI backend for Flask
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import uuid

app = Flask(__name__)
app.secret_key = 'secret-key'

UPLOAD_FOLDER = 'static/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ------------------ DATA PREPROCESSING ------------------
def preprocess_data(df):
    df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
    df['DayOfWeek'] = pd.to_datetime(df['Date'], errors='coerce').dt.dayofweek

    df.dropna(subset=['Hour', 'DayOfWeek', 'Location', 'Severity'], inplace=True)

    df['LocationCode'] = df['Location'].astype('category').cat.codes
    df['Severity'] = df['Severity'].str.title()
    df['HighRisk'] = df['Severity'].apply(lambda x: 1 if x == 'High' else 0)

    return df


# ------------------ VISUALIZATIONS ------------------
def generate_visualizations(df, session_id):
    img1 = f"{session_id}_location.png"
    img2 = f"{session_id}_hours.png"

    # Chart 1: Location vs Severity
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Location', hue='Severity')
    plt.xticks(rotation=45)
    plt.title("Incidents by Location and Severity")
    plt.tight_layout()
    plt.savefig(os.path.join(UPLOAD_FOLDER, img1))
    plt.close()

    # Chart 2: Hourly distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Hour'], bins=24, kde=True)
    plt.title("Incidents by Hour")
    plt.xlabel("Hour of Day")
    plt.tight_layout()
    plt.savefig(os.path.join(UPLOAD_FOLDER, img2))
    plt.close()

    return img1, img2


# ------------------ ML MODELS ------------------
def train_models(df):
    X = df[['Hour', 'DayOfWeek', 'LocationCode']]
    y = df['HighRisk']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    log_model = LogisticRegression(max_iter=1000)
    tree_model = DecisionTreeClassifier(random_state=42)

    log_model.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)

    log_pred = log_model.predict(X_test)
    tree_pred = tree_model.predict(X_test)

    log_report = classification_report(y_test, log_pred, output_dict=True)
    tree_report = classification_report(y_test, tree_pred, output_dict=True)

    return log_report, tree_report


# ------------------ RECOMMENDATIONS ------------------
def generate_recommendations(log_report, tree_report):
    recs = []

    log_recall = log_report.get('1', {}).get('recall', 0)
    tree_precision = tree_report.get('1', {}).get('precision', 0)

    if log_recall < 0.7:
        recs.append("Improve detection of high-risk incidents by enhancing logging practices.")

    if tree_precision < 0.7:
        recs.append("Reduce false alarms with clearer severity classification criteria.")

    if not recs:
        recs.append("Your safety incident model is performing well. Continue regular audits.")

    return recs


# ------------------ ROUTES ------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('file')

    if not file:
        return "No file uploaded", 400

    try:
        df = pd.read_csv(file)

        required_cols = ['Date', 'Time', 'Location', 'Severity']
        if not all(col in df.columns for col in required_cols):
            return f"Missing required columns: {required_cols}", 400

        session_id = str(uuid.uuid4())

        df = preprocess_data(df)
        img1, img2 = generate_visualizations(df, session_id)

        log_report, tree_report = train_models(df)
        recommendations = generate_recommendations(log_report, tree_report)

        session['img1'] = img1
        session['img2'] = img2
        session['recommendations'] = recommendations

        return redirect(url_for('results'))

    except Exception as e:
        return f"Error processing file: {e}", 500


@app.route('/results')
def results():
    return render_template(
        'results.html',
        img1=session.get('img1'),
        img2=session.get('img2'),
        recommendations=session.get('recommendations')
    )


# ------------------ RUN ------------------
if __name__ == '__main__':
    app.run(debug=True)
