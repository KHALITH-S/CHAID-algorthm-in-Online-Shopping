from flask import Flask, render_template, request, redirect, url_for, flash, session
import matplotlib.pyplot as plt  # For plotting
import io
import base64
from sklearn.metrics import accuracy_score, f1_score, precision_score, mean_squared_error

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import random
from faker import Faker
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with a strong secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

with app.app_context():
    db.create_all()

def generate_synthetic_data(num_rows=1000):
    fake = Faker()
    data = []
    for _ in range(num_rows):
        data_collection_method = random.choice(['Web Scraping', 'API Integration', 'Database Dump', 'Surveys'])
        festive_season = random.choice(['Diwali', 'Christmas', 'Black Friday', 'New Year', 'None'])
        offer_type = random.choice(['Discount', 'Cashback', 'Free Gift', 'Bundle Deal', 'None'])
        offer_percentage = random.randint(5, 50)
        pricing_strategy = random.choice(['Dynamic Pricing', 'Fixed Pricing', 'Competitive Pricing'])
        price_fluctuation = round(random.uniform(-0.1, 0.1), 2)
        customer_segment = random.choice(['Price-Sensitive', 'Brand-Loyal', 'Value-Seekers', 'Luxury Buyers'])
        purchase_decision = random.choice(['Yes', 'No'])
        data.append([data_collection_method, festive_season, offer_type, offer_percentage,
                     pricing_strategy, price_fluctuation, customer_segment, purchase_decision])
    df = pd.DataFrame(data, columns=['DataCollectionMethod', 'FestiveSeason', 'OfferType',
                                    'OfferPercentage', 'PricingStrategy', 'PriceFluctuation',
                                    'CustomerSegment', 'PurchaseDecision'])
    return df

# CHAID Analysis Function (using DecisionTreeClassifier as a proxy)
def perform_chaid_analysis(df):
    X = df.drop('PurchaseDecision', axis=1)
    y = df['PurchaseDecision']

    # One-hot encode categorical features
    X = pd.get_dummies(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the DecisionTreeClassifier (as a CHAID proxy)
    model = DecisionTreeClassifier(criterion='entropy', max_depth=4)  # Adjust max_depth as needed
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label='Yes')  # F1-score for 'Yes' class
    precision = precision_score(y_test, y_pred, pos_label='Yes')

    # RMSE (Not ideal for classification, but included for demonstration)
    try:
        rmse = mean_squared_error(y_test.map({'Yes': 1, 'No': 0}), [1 if p == 'Yes' else 0 for p in y_pred], squared=False)
    except Exception as e:  # Catch potential errors (e.g., if y_test has unexpected values)
        rmse = f"N/A (Error: {e})"

    # Feature Importances
    feature_importances = model.feature_importances_
    feature_importance_list = [{'feature': feature, 'importance': importance}
                               for feature, importance in zip(X.columns, feature_importances)]

    # Graphs
    graphs = {}

    # 1. Price Fluctuation
    graphs['price_fluctuation'] = create_plot(df.index, df['PriceFluctuation'], 'Data Point', 'Price Fluctuation', 'Price Fluctuation Over Data Points', 'line')

    # 2. Purchase Decision
    graphs['purchase_decision'] = create_plot(df['PurchaseDecision'].value_counts().index, df['PurchaseDecision'].value_counts().values, 'Purchase Decision', 'Count', 'Purchase Decision Distribution', 'bar')

    # 3. Offer Percentage
    graphs['offer_percentage'] = create_plot(df['OfferPercentage'].value_counts().index, df['OfferPercentage'].value_counts().values, 'Offer Percentage', 'Count', 'Offer Percentage Distribution', 'bar')  # Or 'hist'

    # 4. Pricing Strategy
    graphs['pricing_strategy'] = create_plot(df['PricingStrategy'].value_counts().index, df['PricingStrategy'].value_counts().values, 'Pricing Strategy', 'Count', 'Pricing Strategy Distribution', 'bar')

    return feature_importance_list, accuracy, f1, precision, rmse, graphs

def create_plot(x, y, xlabel, ylabel, title, plot_type):
    plt.figure(figsize=(8, 6))
    if plot_type == 'line':
        plt.plot(x, y, marker='o', linestyle='-')
    elif plot_type == 'bar':
        plt.bar(x, y)
    elif plot_type == 'hist':
        plt.hist(y)  # For histogram
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    return get_graph()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please login.')
            return redirect(url_for('login'))

        new_user = User()
        new_user.username = username
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful. Please login.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['is_admin'] = user.is_admin
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('is_admin', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    return render_template('dashboard.html')

def get_graph():
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.clf()  # Clear the plot for the next graph
    return f'data:image/png;base64,{graph_url}'

@app.route('/analyze', methods=['POST'])
def analyze():
    if not session.get('user_id'):
        return redirect(url_for('login'))

    num_rows = int(request.form.get('num_rows', 1000))
    df = generate_synthetic_data(num_rows)
    results, accuracy, f1, precision, rmse, graphs = perform_chaid_analysis(df)
    return render_template('results.html', results=results, accuracy=accuracy, f1=f1, 
                           precision=precision, rmse=rmse, graphs=graphs)

@app.route('/admin')
def admin():
    if not session.get('user_id') or not session.get('is_admin'):
        return redirect(url_for('login'))

    users = User.query.all()
    return render_template('admin.html', users=users)

if __name__ == '__main__':
    app.run(debug=True)