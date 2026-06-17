import random
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "trading_secret_key"

# Mock Database holding both usernames and emails
users_db = {
    "trader123": {"email": "trader@example.com", "password": "password123"},
    "admin": {"email": "admin@dss.com", "password": "securepassword"}
}

# Mock market data dictionary for different assets
MARKET_DATA = {
    "XAUUSD": {"asset": "XAUUSD (Gold)", "action": "BUY", "confidence": 88.5, "uncertainty_limit": 0.15, "target_exit": 2045.50},
    "FBMKLCI": {"asset": "FBM KLCI", "action": "HOLD", "confidence": 52.1, "uncertainty_limit": 0.48, "target_exit": 1610.20},
    "BTCUSD": {"asset": "Bitcoin", "action": "SELL", "confidence": 74.3, "uncertainty_limit": 0.22, "target_exit": 64200.00},
    "EURUSD": {"asset": "EURUSD", "action": "BUY", "confidence": 69.8, "uncertainty_limit": 0.18, "target_exit": 1.0920}
}

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login_input = request.form.get('login_input').strip()
        password = request.form.get('password')
        
        # Check database if input matches username or email key
        authenticated = False
        for username, data in users_db.items():
            if (login_input == username or login_input == data['email']) and data['password'] == password:
                authenticated = True
                break
                
        if authenticated:
            return redirect(url_for('dashboard'))
        
        flash("Invalid username or password", "danger")
    return render_template('login.html', type="login")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email').strip()
        
        # 50% chance to randomly throw a duplicate error, or check explicit mock DB match
        if random.choice([True, False]) or any(data['email'] == email for data in users_db.values()):
            flash("Registration failed: Email already exists in our system.", "danger")
            return render_template('login.html', type="register")
        
        flash("Account created successfully! Proceed to Login.", "success")
        return redirect(url_for('login'))
        
    return render_template('login.html', type="register")

@app.route('/dashboard')
def dashboard():
    selected_asset = request.args.get('asset', 'XAUUSD')
    data = MARKET_DATA.get(selected_asset, MARKET_DATA['XAUUSD'])
    return render_template('dashboard.html', data=data, current_asset=selected_asset)

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/billing')
def billing():
    return render_template('billing.html')

if __name__ == '__main__':
    app.run(debug=True)