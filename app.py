from flask import Flask, request, jsonify, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from flask_restx import Api, Resource, fields
from apscheduler.schedulers.background import BackgroundScheduler
import joblib
import os
import math
import logging
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load env variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
api = Api(app, version="1.0", title="GoalStats API",
          description="Advanced Football Prediction API with multiple markets and ML",
          doc="/docs")  # Swagger UI

# Rate limiter
limiter = Limiter(app, key_func=get_remote_address, default_limits=["100 per hour"])

# Cache config
cache = Cache(app, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300})

# API Keys (validate on startup)
FOOTBALL_API_TOKEN = os.getenv("FOOTBALL_API_TOKEN")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
missing_keys = []
if not FOOTBALL_API_TOKEN: missing_keys.append("FOOTBALL_API_TOKEN")
if not ODDS_API_KEY: missing_keys.append("ODDS_API_KEY")
if not WEATHER_API_KEY: missing_keys.append("WEATHER_API_KEY")
if missing_keys:
    raise EnvironmentError(f"Missing required API keys: {', '.join(missing_keys)}")

# Headers
football_headers = {
    "X-RapidAPI-Key": FOOTBALL_API_TOKEN,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}

# Directory to save/load models
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Utility functions ---

def poisson_prob(lmbda, k):
    return (lmbda ** k * math.exp(-lmbda)) / math.factorial(k)

def kelly_criterion(prob, odds):
    if odds <= 1: return 0
    return max(0, (prob * (odds - 1) - (1 - prob)) / (odds - 1))

def implied_prob(odds):
    return 1 / odds if odds > 0 else 0

def validate_args(*args):
    for arg in args:
        if not request.args.get(arg):
            return jsonify({"error": f"Missing parameter: {arg}"}), 400
    return None

def load_training_data(market):
    """
    Placeholder function to simulate loading historical labeled data
    for different markets. Return X (features), y (labels).
    """
    # You must replace this with actual data loading code
    logging.info(f"Loading training data for market '{market}'")
    if market == "btts":
        # Simulate BTTS data: features= [home_avg_goals, away_avg_goals], label=0/1
        X = np.array([[1.2,1.3],[2.0,0.5],[0.5,0.7],[1.8,1.9],[0.3,0.2]])
        y = np.array([1,1,0,1,0])
    elif market == "over25":
        X = np.array([[1.2,1.3],[2.0,0.5],[0.5,0.7],[1.8,1.9],[0.3,0.2]])
        y = np.array([1,1,0,1,0])
    else:  # Default for 1X2
        X = np.array([[1.2,1.3],[2.0,0.5],[0.5,0.7],[1.8,1.9],[0.3,0.2]])
        y = np.array([0,2,1,0,1])  # 0=home win, 1=draw, 2=away win
    return X, y

def train_model_for_market(market):
    """
    Train and save a sklearn logistic regression model for the given market.
    """
    X, y = load_training_data(market)
    logging.info(f"Training model for {market} with data X shape {X.shape} and y shape {y.shape}")

    # For simplicity, binary classification for btts and over25, multiclass for 1X2
    if market in ["btts", "over25"]:
        model = LogisticRegression()
    else:
        model = LogisticRegression(multi_class='multinomial', max_iter=500)

    model.fit(X, y)
    model_path = os.path.join(MODEL_DIR, f"{market}_model.joblib")
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")
    return model_path

def predict_market(market, features):
    """
    Load model and predict probabilities/stakes for given market and features
    features: list or np.array of features
    """
    model_path = os.path.join(MODEL_DIR, f"{market}_model.joblib")
    if not os.path.exists(model_path):
        # Auto-train if model missing
        train_model_for_market(market)

    model = joblib.load(model_path)
    features = np.array(features).reshape(1, -1)
    proba = model.predict_proba(features)[0]

    logging.info(f"Predicted probabilities for {market}: {proba}")

    # Example response formatting per market:
    if market == "btts":
        # proba: [No, Yes]
        value_bet = "Yes" if proba[1] * 2.0 > 1 else "No"  # example odds=2.0
        stake = kelly_criterion(proba[1], 2.0)
        return {
            "market": market,
            "probability_no": round(proba[0], 3),
            "probability_yes": round(proba[1], 3),
            "value_bet": value_bet,
            "recommended_stake": round(stake, 3)
        }
    elif market == "over25":
        # proba: [No, Yes]
        value_bet = "Yes" if proba[1] * 1.8 > 1 else "No"
        stake = kelly_criterion(proba[1], 1.8)
        return {
            "market": market,
            "probability_no": round(proba[0], 3),
            "probability_yes": round(proba[1], 3),
            "value_bet": value_bet,
            "recommended_stake": round(stake, 3)
        }
    else:
        # multiclass for 1X2
        outcomes = ["Home Win", "Draw", "Away Win"]
        odds = [2.1, 3.2, 3.3]  # Ideally fetch from real odds input
        value_bets = [outcomes[i] for i in range(3) if proba[i] * odds[i] > 1]
        stakes = [round(kelly_criterion(proba[i], odds[i]), 3) for i in range(3)]
        return {
            "market": market,
            "probabilities": {outcomes[i]: round(proba[i], 3) for i in range(3)},
            "value_bets": value_bets,
            "recommended_stakes": stakes
        }

# --- Scheduled Background Tasks ---

scheduler = BackgroundScheduler()

def scheduled_data_refresh():
    logging.info("Scheduled background task running: refreshing cached data or retraining models...")
    # Implement actual refresh or retrain here

scheduler.add_job(scheduled_data_refresh, 'interval', minutes=60)
scheduler.start()

# --- API Models for Swagger ---

prediction_model = api.model('Prediction', {
    'market': fields.String(required=True, description='Prediction market e.g. 1x2, btts, over25'),
    'home_avg': fields.Float(required=True, description='Average home goals scored'),
    'away_avg': fields.Float(required=True, description='Average away goals scored'),
    'home_odds': fields.Float(required=False, description='Decimal odds home win'),
    'draw_odds': fields.Float(required=False, description='Decimal odds draw'),
    'away_odds': fields.Float(required=False, description='Decimal odds away win')
})

# --- Routes ---

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')

@app.before_request
def log_route_access():
    logging.info(f"Incoming request: {request.method} {request.path}")

@app.errorhandler(404)
def fallback_404(e):
    logging.warning(f"404 Not Found: {request.path}")
    return jsonify({"error": "Route not found", "path": request.path}), 404

@app.route("/")
def home():
    return jsonify({"message": "Welcome to GoalStats API! Visit /docs for API documentation."})

# --- Training Endpoint ---

@app.route("/api/train_model", methods=["POST"])
@limiter.limit("5/hour")
def train_model():
    market = request.json.get("market")
    if not market:
        return jsonify({"error": "Missing 'market' in JSON body"}), 400
    try:
        model_path = train_model_for_market(market)
        return jsonify({"message": f"Model trained and saved at {model_path}", "market": market})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Prediction Endpoint ---

@app.route("/api/predict", methods=["GET"])
@limiter.limit("50/hour")
@cache.cached(timeout=60, query_string=True)
def predict():
    market = request.args.get("market")
    home_avg = request.args.get("home_avg", type=float)
    away_avg = request.args.get("away_avg", type=float)
    if not market or home_avg is None or away_avg is None:
        return jsonify({"error": "Missing required parameters: market, home_avg, away_avg"}), 400

    features = [home_avg, away_avg]
    try:
        result = predict_market(market, features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Backtesting Endpoint ---

@app.route("/api/backtest", methods=["POST"])
@limiter.limit("2/hour")
def backtest():
    data = request.json
    market = data.get("market")
    bets = data.get("bets")  # list of {match_id, predicted_outcome, odds, actual_outcome}

    if not market or not bets:
        return jsonify({"error": "Missing 'market' or 'bets' in request body"}), 400

    # Simulate backtesting
    total_stake = 0
    total_return = 0
    wins = 0
    for bet in bets:
        stake = bet.get("stake", 1)
        total_stake += stake
        predicted = bet.get("predicted_outcome")
        actual = bet.get("actual_outcome")
        odds = bet.get("odds", 1)
        if predicted == actual:
            total_return += stake * odds
            wins += 1

    roi = ((total_return - total_stake) / total_stake) if total_stake > 0 else 0
    win_rate = (wins / len(bets)) if bets else 0

    return jsonify({
        "market": market,
        "total_bets": len(bets),
        "wins": wins,
        "win_rate": round(win_rate, 3),
        "roi": round(roi, 3),
        "total_stake": total_stake,
        "total_return": round(total_return, 3)
    })

# --- API Namespace for Swagger ---

ns = api.namespace('football', description='Football prediction operations')

@ns.route('/predict')
class FootballPredict(Resource):
    @api.expect(prediction_model)
    def post(self):
        json_data = api.payload
        market = json_data.get("market")
        home_avg = json_data.get("home_avg")
        away_avg = json_data.get("away_avg")

        if not market or home_avg is None or away_avg is None:
            return {"error": "Missing parameters"}, 400

        try:
            features = [home_avg, away_avg]
            result = predict_market(market, features)
            return result
        except Exception as e:
            return {"error": str(e)}, 500

# --- Run ---

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
