from datetime import datetime
from flask import send_file
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)
from flask import abort
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib
from flask_sqlalchemy import SQLAlchemy
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import uuid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import jsonify, send_file
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt



app = Flask(__name__)
app.secret_key = 'secret-key'
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
rf_model = None
chart1_global = None
chart2_global = None
chart3_global = None
chart4_global = None
chart5_global = None

from datetime import datetime

class User(UserMixin, db.Model):

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(
        db.String(50),
        unique=True,
        nullable=False
    )
    email = db.Column(
        db.String(100),
        unique=True,
        nullable=False
    )
    password = db.Column(
        db.String(200),
        nullable=False
    )

    role = db.Column(
        db.String(20),
        default='Employee'
    )

    analyses_count = db.Column(
        db.Integer,
        default=0
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )
class Incident(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    date = db.Column(db.String(20))

    time = db.Column(db.String(20))

    location = db.Column(db.String(50))

    severity = db.Column(db.String(20))

UPLOAD_FOLDER = 'static/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
def preprocess_data(df):
    df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
    df['DayOfWeek'] = pd.to_datetime(df['Date'], errors='coerce').dt.dayofweek

    df.dropna(subset=['Hour', 'DayOfWeek', 'Location', 'Severity'], inplace=True)

    df['LocationCode'] = df['Location'].astype('category').cat.codes
    df['Severity'] = df['Severity'].str.title()
    df['HighRisk'] = df['Severity'].apply(lambda x: 1 if x == 'High' else 0)

    return df

def generate_visualizations(df):

    location_data = (
        df.groupby(['Location', 'Severity'])
        .size()
        .reset_index(name='Count')
    )

    fig1 = px.bar(
        location_data,
        x='Location',
        y='Count',
        color='Severity',
        title='Incidents by Location and Severity',
        barmode='group'
    )
    fig2 = px.histogram(
        df,
        x='Hour',
        nbins=24,
        title='Incident Distribution by Hour'
    )
    fig3 = px.pie(
        df,
        names='Severity',
        title='Severity Distribution'
    )
    day_data = (
        df.groupby('DayOfWeek')
        .size()
        .reset_index(name='Count')
    )

    fig4 = px.line(
        day_data,
        x='DayOfWeek',
        y='Count',
        title='Incidents by Day'
    )
    fig5 = px.scatter(
        df,
        x='Hour',
        y='DayOfWeek',
        color='Severity',
        title='Risk Pattern'
    )
    fig1.write_image("chart1.png")
    fig2.write_image("chart2.png")
    fig3.write_image("chart3.png")
    fig4.write_image("chart4.png")
    fig5.write_image("chart5.png")

    chart1 = fig1.to_html(full_html=False)
    chart2 = fig2.to_html(full_html=False)
    chart3 = fig3.to_html(full_html=False)
    chart4 = fig4.to_html(full_html=False)
    chart5 = fig5.to_html(full_html=False)

    return chart1, chart2, chart3, chart4, chart5
def train_models(df):
    global rf_model
    X = df[['Hour', 'DayOfWeek', 'LocationCode']]
    y = df['HighRisk']
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
   )

    rf_model = RandomForestClassifier()
    log_model = LogisticRegression(max_iter=1000)
    tree_model = DecisionTreeClassifier(random_state=42)

    rf_model.fit(X_train, y_train)
    log_model.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    log_pred = log_model.predict(X_test)
    tree_pred = tree_model.predict(X_test)

    print("y_test =", y_test.values)
    print("rf_pred =", rf_pred)
    print("log_pred =", log_pred)
    print("tree_pred =", tree_pred)

    log_report = classification_report(y_test, log_pred, output_dict=True)
    tree_report = classification_report(y_test, tree_pred, output_dict=True)
    log_accuracy = round(
        accuracy_score(y_test, log_pred) * 100,
        2
    )

    tree_accuracy = round(
        accuracy_score(y_test, tree_pred) * 100,
        2
    )

    rf_accuracy = round(
        accuracy_score(y_test, rf_pred) * 100,
        2
    )
    print("Logistic Accuracy =", log_accuracy)
    print("Decision Tree Accuracy =", tree_accuracy)
    print("Random Forest Accuracy =", rf_accuracy)

    return (
        log_report,
        tree_report,
        rf_model,
        log_accuracy,
        tree_accuracy,
        rf_accuracy
    )
    

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

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':

        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        existing_user = User.query.filter(
            (User.username == username) |
            (User.email == email)
        ).first()

        if existing_user:
            return "Username or Email already exists!"

        hashed_password = bcrypt.generate_password_hash(
            password
        ).decode('utf-8')

        user = User(
            username=username,
            email=email,
            password=hashed_password,
            role=role
        )
        db.session.add(user)
        db.session.commit()

        return redirect(url_for('login'))


    return render_template('register.html')
@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):

          login_user(user)

          return redirect(url_for('index'))

    flash("Invalid email or password","danger")

    return render_template('login.html')
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():

    if request.method == 'POST':

        email = request.form['email']

        user = User.query.filter_by(email=email).first()

        if user:

            return redirect(url_for('reset_password', user_id=user.id))

        flash("Email not found")

    return render_template("forgot_password.html")
@app.route('/reset_password/<int:user_id>', methods=['GET','POST'])
def reset_password(user_id):

    user = User.query.get_or_404(user_id)

    if request.method == 'POST':

        password = request.form['password']

        hashed = bcrypt.generate_password_hash(password).decode('utf-8')

        user.password = hashed

        db.session.commit()

        flash("Password Updated Successfully")

        return redirect(url_for('login'))

    return render_template("reset_password.html")
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():

    if request.method == 'POST':

        file = request.files.get('file')

        if not file:
            flash("Please select a CSV file.", "danger")
            return redirect(url_for('upload'))

        df = pd.read_csv(file)

        required_cols = ['Date', 'Time', 'Location', 'Severity']
        if not all(col in df.columns for col in required_cols):
            return f"Missing required columns: {required_cols}", 400

        df = preprocess_data(df)

        for _, row in df.iterrows():
            incident = Incident(
                date=str(row['Date']),
                time=str(row['Time']),
                location=row['Location'],
                severity=row['Severity']
            )
            db.session.add(incident)

        db.session.commit()

        chart1, chart2, chart3, chart4, chart5 = generate_visualizations(df)

        global rf_model

        log_report, tree_report, rf_model, log_accuracy, tree_accuracy, rf_accuracy = train_models(df)

        recommendations = generate_recommendations(log_report, tree_report)

        session['total'] = len(df)
        session['high'] = len(df[df['Severity'] == 'High'])
        session['locations'] = df['Location'].nunique()

        global chart1_global, chart2_global, chart3_global, chart4_global, chart5_global

        chart1_global = chart1
        chart2_global = chart2
        chart3_global = chart3
        chart4_global = chart4
        chart5_global = chart5

        session['recommendations'] = recommendations
        session['log_accuracy'] = log_accuracy
        session['tree_accuracy'] = tree_accuracy
        session['rf_accuracy'] = rf_accuracy

        current_user.analyses_count += 1
        db.session.commit()

        return redirect(url_for('results'))

    return render_template('upload.html')
@app.route('/')
@login_required
def index():
    return render_template('index.html')

# @app.route('/analyze', methods=['POST'])
# def analyze():
#     file = request.files.get('file')

#     if not file:
#         return "No file uploaded", 400

#     try:
#         df = pd.read_csv(file)

#         required_cols = ['Date', 'Time', 'Location', 'Severity']
#         if not all(col in df.columns for col in required_cols):
#             return f"Missing required columns: {required_cols}", 400

#         session_id = str(uuid.uuid4())

#         df = preprocess_data(df)
#         for _, row in df.iterrows():

#            incident = Incident(
#                 date=str(row['Date']),
#                 time=str(row['Time']),
#                 location=row['Location'],
#                 severity=row['Severity']
#             )
#            db.session.add(incident)
#         db.session.commit()
#         chart1, chart2, chart3, chart4, chart5 = generate_visualizations(df)

#         global rf_model

#         log_report, tree_report, rf_model, log_accuracy, tree_accuracy, rf_accuracy = train_models(df)
#         recommendations = generate_recommendations(log_report, tree_report)

#         session['total'] = len(df)
#         session['high'] = len(df[df['Severity'] == 'High'])
#         session['locations'] = df['Location'].nunique()
#         global  chart1_global, chart2_global, chart3_global, chart4_global, chart5_global
#         chart1_global = chart1
#         chart2_global = chart2
#         chart3_global = chart3
#         chart4_global = chart4
#         chart5_global = chart5
#         session['recommendations'] = recommendations
#         session['log_accuracy'] = log_accuracy
#         session['tree_accuracy'] = tree_accuracy
#         session['rf_accuracy'] = rf_accuracy
#         current_user.analyses_count += 1
#         db.session.commit()
#         return redirect(url_for('results'))
#     except Exception as e:
#         return f"Error processing file: {e}", 500
@app.route('/results')
@login_required
def results():

    return render_template(
        "results.html",
        log_accuracy=session.get('log_accuracy'),
        tree_accuracy=session.get('tree_accuracy'),
        rf_accuracy=session.get('rf_accuracy'),
        chart1=chart1_global,
        chart2=chart2_global,
        chart3=chart3_global,
        chart4=chart4_global,
        chart5=chart5_global,
        total=session.get('total'),
        high=session.get('high'),
        locations=session.get('locations'),
        # recommendations=session.get(
        #     'recommendations',
        #     [
        #         "Conduct regular safety audits",
        #         "Increase PPE compliance",
        #         "Provide employee training",
        #         "Monitor high-risk locations",
        #         "Improve incident reporting"
        #     ]
        # )
    )
@app.route('/predict')
@login_required
def predict_page():
    return render_template("predict.html")
@app.route('/risk',methods=['POST'])
def risk():
    global rf_model
    if rf_model is None:
        return render_template("predict.html", result="Error: Model not trained")
    hour = int(request.form['hour'])
    day = int(request.form['day'])
    location = int(request.form['location'])

    prediction = rf_model.predict([[hour,day,location]])

    if prediction[0] == 1:
        result = "High Risk"
    else:
        result = "Low Risk"
    session['prediction_result'] = result

    return render_template("predict.html",result=result)
#@app.route('/api/incidents')
# def api_incidents():
#     df = session.get('df')
    
#     if df is None:
#         return jsonify({'error': 'No data available'}), 400

#     return jsonify(df.to_dict(orient='records'))
@app.route('/api/predict',methods=['POST'])
def api_predict():
    global rf_model
    
    if rf_model is None:
        return jsonify({'error': 'Model not trained'}), 400
    
    data = request.get_json()

    hour = int(data['hour'])
    day = int(data['day'])
    location = int(data['location'])

    prediction = rf_model.predict([[hour,day,location]])

    risk = "High Risk" if prediction[0] == 1 else "Low Risk"

    return jsonify({'risk': risk})
@app.route('/api/incidents', methods=['GET'])
def get_incidents():

    incidents = Incident.query.all()

    data = []

    for i in incidents:

        data.append({
            'id': i.id,
            'date': i.date,
            'time': i.time,
            'location': i.location,
            'severity': i.severity
        })

    return jsonify(data)
@app.route('/api/incidents', methods=['POST'])
def create_incident():

    data = request.get_json()

    incident = Incident(
        date=data['date'],
        time=data['time'],
        location=data['location'],
        severity=data['severity']
    )

    db.session.add(incident)
    db.session.commit()

    return jsonify({
        "message":"Incident added successfully"
    })
@app.route('/api/incidents/<int:id>', methods=['PUT'])
def update_incident(id):

    incident = Incident.query.get_or_404(id)

    data = request.get_json()

    incident.date = data['date']
    incident.time = data['time']
    incident.location = data['location']
    incident.severity = data['severity']

    db.session.commit()

    return jsonify({
        "message":"Incident updated"
    })
@app.route('/api/incidents/<int:id>', methods=['DELETE'])
def delete_incident(id):

    incident = Incident.query.get_or_404(id)

    db.session.delete(incident)

    db.session.commit()

    return jsonify({
        "message":"Incident deleted"
    })
with app.app_context():
    db.create_all()
@app.route('/download_pdf')
def download_pdf():

    doc = SimpleDocTemplate("report.pdf")

    styles = getSampleStyleSheet()
    elements = []

    elements.append(
        Paragraph("RiskLens Analysis Report", styles['Title'])
    )

    elements.append(Spacer(1,20))

    data = [
        ["Total Incidents", session.get('total')],
        ["High Risk Incidents", session.get('high')],
        ["Locations", session.get('locations')]
    ]

    table = Table(data)

    table.setStyle(
        TableStyle([
            ('BACKGROUND',(0,0),(-1,-1),colors.lightgrey),
            ('BOX',(0,0),(-1,-1),1,colors.black),
            ('GRID',(0,0),(-1,-1),1,colors.black)
        ])
    )

    elements.append(table)

    doc.build(elements)

    return send_file(
        "report.pdf",
        as_attachment=True
    )
@app.route('/download_csv')
def download_csv():

    df = pd.DataFrame({

        'Metric':[
            'Total Incidents',
            'High Risk Incidents',
            'Locations'
        ],

        'Value':[
            session.get('total'),
            session.get('high'),
            session.get('locations')
        ]

    })

    df.to_csv(
        'report.csv',
        index=False
    )

    return send_file(
        'report.csv',
        as_attachment=True
    )
@app.route('/download_report')
def download_report():

    doc = SimpleDocTemplate("RiskLens_Report.pdf")

    styles = getSampleStyleSheet()
    elements = []

    # Title
    title = Paragraph(
        "RiskLens Safety Analysis Report",
        styles['Title']
    )

    elements.append(title)
    elements.append(Spacer(1,20))

    # Date and Time
    current_time = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    elements.append(
        Paragraph(
            f"Generated on: {current_time}",
            styles['Normal']
        )
    )

    elements.append(Spacer(1,20))

    # Dashboard statistics
    data = [

        ["Total Incidents",
         session.get('total')],

        ["High Risk Incidents",
         session.get('high')],

        ["Locations",
         session.get('locations')],

        ["Logistic Regression Accuracy",
         str(session.get('log_accuracy'))],

        ["Decision Tree Accuracy",
         str(session.get('tree_accuracy'))],

        ["Random Forest Accuracy",
         str(session.get('rf_accuracy'))]
    ]

    table = Table(data)

    table.setStyle(

        TableStyle([

            ('BACKGROUND',
             (0,0),(-1,-1),
             colors.lightgrey),

            ('BOX',
             (0,0),(-1,-1),
             1,
             colors.black),

            ('GRID',
             (0,0),(-1,-1),
             1,
             colors.black)

        ])

    )

    elements.append(table)

    elements.append(Spacer(1,25))

    # Recommendations

    elements.append(
        Paragraph(
            "Safety Recommendations",
            styles['Heading2']
        )
    )
    elements.append(
    Image(
        "chart1.png",
        width=5*inch,
        height=3*inch
    )
)
    elements.append(Spacer(1,20))

    elements.append(
        Image(
            "chart2.png",
            width=5*inch,
            height=3*inch
        )
    )
    elements.append(Spacer(1,20))

    elements.append(
        Image(
            "chart3.png",
            width=5*inch,
            height=3*inch
        )
    )
    elements.append(Spacer(1,20))

    elements.append(
        Image(
            "chart4.png",
            width=5*inch,
            height=3*inch
        )
    )
    elements.append(Spacer(1,20))

    elements.append(
        Image(
            "chart5.png",
            width=5*inch,
            height=3*inch
        )
    )
    recommendations = session.get(
        'recommendations',
        []
    )

    for rec in recommendations:

        elements.append(
            Paragraph(
                "• " + rec,
                styles['Normal']
            )
        )

    elements.append(Spacer(1,25))

    # Risk prediction result

    risk_result = session.get(
        'prediction_result',
        "Not Available"
    )

    elements.append(
        Paragraph(
            "Predicted Risk: " + risk_result,
            styles['Heading2']
        )
    )

    elements.append(Spacer(1,20))

    # Charts

    try:

        elements.append(
            Paragraph(
                "Charts",
                styles['Heading2']
            )
        )

        elements.append(
            Image(
                os.path.join(
                    UPLOAD_FOLDER,
                    session['img1']
                ),
                width=5*inch,
                height=3*inch
            )
        )

        elements.append(Spacer(1,20))

        elements.append(
            Image(
                os.path.join(
                    UPLOAD_FOLDER,
                    session['img2']
                ),
                width=5*inch,
                height=3*inch
            )
        )

    except:
        pass

    doc.build(elements)

    return send_file(
        "RiskLens_Report.pdf",
        as_attachment=True
    )
@app.route('/profile')
@login_required
def profile():

    created_date = current_user.created_at.strftime("%d-%m-%Y")

    return render_template(
        'profile.html',
        created_date=created_date
    )
@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():

    query = Incident.query

    if request.method == 'POST':

        date = request.form['date']
        location = request.form['location']
        severity = request.form['severity']

        if date:
            query = query.filter(
                Incident.date.contains(date)
            )

        if location:
            query = query.filter(
                Incident.location.contains(location)
            )

        if severity:
            query = query.filter(
                Incident.severity.contains(severity)
            )

    incidents = query.all()

    return render_template(
        'search.html',
        incidents=incidents
    )
@app.route('/admin')
@login_required
def admin():
    admin_required()
    total_users = User.query.count()

    total_incidents = Incident.query.count()

    high_risk_incidents = Incident.query.filter_by(
        severity='High'
    ).count()

    dangerous_location = (
        db.session.query(
            Incident.location,
            db.func.count(Incident.id)
        )
        .filter(Incident.severity == 'High')
        .group_by(Incident.location)
        .order_by(
            db.func.count(Incident.id).desc()
        )
        .first()
    )

    if dangerous_location:
        dangerous_location = dangerous_location[0]
    else:
        dangerous_location = "No Data"
    users = User.query.all()

    return render_template(
        'admin.html',
        users=users,
        total_users=total_users,
        total_incidents=total_incidents,
        high_risk=high_risk_incidents,
        dangerous_location=dangerous_location
    )
@app.route('/test')
@app.route('/api/highrisk')
def high_risk():

    incidents = Incident.query.filter_by(
        severity='High'
    ).all()

    data = []

    for i in incidents:

        data.append({

            'id': i.id,
            'date': i.date,
            'time': i.time,
            'location': i.location,
            'severity': i.severity

        })

    return jsonify(data)
@app.route('/api/location/<location>')
def location_incidents(location):

    incidents = Incident.query.filter_by(
        location=location
    ).all()

    data = []

    for i in incidents:

        data.append({

            'id': i.id,
            'date': i.date,
            'time': i.time,
            'location': i.location,
            'severity': i.severity

        })

    return jsonify(data)
@app.route('/api/severity/<severity>')
def severity_incidents(severity):

    incidents = Incident.query.filter_by(
        severity=severity
    ).all()

    data = []

    for i in incidents:

        data.append({

            'id': i.id,
            'date': i.date,
            'time': i.time,
            'location': i.location,
            'severity': i.severity

        })

    return jsonify(data)
@app.route('/api/date/<date>')
def date_incidents(date):

    incidents = Incident.query.filter_by(
        date=date
    ).all()

    data = []

    for i in incidents:

        data.append({

            'id': i.id,
            'date': i.date,
            'time': i.time,
            'location': i.location,
            'severity': i.severity

        })
    return jsonify(data)

def admin_required():

    if current_user.role != "Admin":
        abort(403)
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    )
