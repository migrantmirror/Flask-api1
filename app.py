from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from flask_cors import CORS
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib
import requests
import logging
import os
import math

# --- Config & Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/hour"],
    storage_uri="memory://"
)
limiter.init_app(app)

cache = Cache(app, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300})

API_TOKEN = os.getenv("FOOTBALL_API_TOKEN")
if not API_TOKEN:
    raise EnvironmentError("Missing FOOTBALL_API_TOKEN")

football_headers = {
    "X-RapidAPI-Key": API_TOKEN,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}

MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Utilities ---

def poissonProb(lmbda, k):
    """Calculate Poisson probability for k goals given average λ"""
    try:
        return (lmbda**k * math.exp(-lmbda)) / math.factorial(k)
    except:
        return 0.0

def predictCorrectScore(home_avg, away_avg, max_goals=5):
    """Generate a grid of score probabilities using Poisson distribution"""
    grid = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poissonProb(home_avg, h) * poissonProb(away_avg, a)
            grid[f"{h}-{a}"] = round(p, 5)
    return grid

def trainAndPredict(market, features):
    """Train and/or load a Logistic Regression model and predict market outcome"""
    model_path = os.path.join(MODEL_DIR, f"{market}.joblib")

    if not os.path.exists(model_path):
        # Fallback data (replace with real match data for production)
        X = np.array([[1, 1], [2, 0], [0, 2], [1.5, 1.5]])
        y = np.array([1, 0, 1, 1]) if market != "btts" else np.array([0, 1, 1, 1])
        clf = LogisticRegression()
        clf.fit(X, y)
        joblib.dump(clf, model_path)
        logging.info(f"Trained and saved new model for {market}")
    else:
        clf = joblib.load(model_path)

    proba = clf.predict_proba([features])[0]
    odds = 2.0
    return {
        "proba": [round(x, 4) for x in proba],
        "value": round(proba[1] * odds > 1, 2)
    }

# --- Routes ---

@app.route("/")
def index():
    return jsonify({"message": "GoalStats AI API is live"})

@app.route("/api/live_matches")
def live_matches():
    try:
        response = requests.get(
            "https://api-football-v1.p.rapidapi.com/v3/fixtures?live=all",
            headers=football_headers
        )
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/team_stats")
def team_stats():
    team_id = request.args.get("team_id")
    if not team_id:
        return jsonify({"error": "Missing team_id"}), 400
    try:
        response = requests.get(
            f"https://api-football-v1.p.rapidapi.com/v3/teams/statistics?team={team_id}&season=2023",
            headers=football_headers
        )
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/correct_score")
@limiter.limit("50/hour")
def correct_score():
    try:
        home_avg = float(request.args.get("home_avg", 1.4))
        away_avg = float(request.args.get("away_avg", 1.1))
        grid = predictCorrectScore(home_avg, away_avg)
        top5 = dict(sorted(grid.items(), key=lambda x: -x[1])[:5])
        return jsonify({"probabilities": top5})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/predict_market")
@limiter.limit("50/hour")
def predict_market():
    try:
        market = request.args.get("market", "btts")
        home_avg = float(request.args.get("home_avg", 1.4))
        away_avg = float(request.args.get("away_avg", 1.1))
        result = trainAndPredict(market, [home_avg, away_avg])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- Run App ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)
