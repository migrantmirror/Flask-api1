from flask import Flask, request, jsonify, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from apscheduler.schedulers.background import BackgroundScheduler
import joblib, os, math, logging, atexit
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
import numpy as np
from flask_cors import CORS
import requests

# Load env variables
load_dotenv()
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# Initialize Limiter without app here
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/hour"],
    storage_uri="memory://"
)

# Attach Limiter to app explicitly
limiter.init_app(app)

cache = Cache(app, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300})

# API keys
FOOTBALL_API_TOKEN = os.getenv("FOOTBALL_API_TOKEN")
if not FOOTBALL_API_TOKEN:
    raise EnvironmentError("Missing FOOTBALL_API_TOKEN")

football_headers = {
    "X-RapidAPI-Key": FOOTBALL_API_TOKEN,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}

MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Utilities ---
def poissonProb(lmbda, k):
    return (lmbda**k * math.exp(-lmbda)) / math.factorial(k)

def predictCorrectScore(home_avg, away_avg, max_goals=5):
    grid = {}
    for h in range(max_goals+1):
        for a in range(max_goals+1):
            p = poissonProb(home_avg, h) * poissonProb(away_avg, a)
            grid[f"{h}-{a}"] = round(p, 5)
    return grid

def trainAndPredict(market, features):
    path = os.path.join(MODEL_DIR, f"{market}.joblib")
    if not os.path.exists(path):
        # Dummy train
        X = np.array([[1,1],[2,0],[0,2]])
        y = np.array([1,0,1]) if market != "btts" else np.array([0,1,1])
        clf = LogisticRegression().fit(X, y)
        joblib.dump(clf, path)
    clf = joblib.load(path)
    proba = clf.predict_proba([features])[0]
    odds = 2.0
    return {"proba": proba.tolist(), "value": proba[1]*odds>1}

# --- Endpoints ---

@app.route("/api/live_matches")
def live_matches():
    try:
        resp = requests.get("https://api-football-v1.p.rapidapi.com/v3/fixtures?live=all", headers=football_headers)
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/team_stats")
def team_stats():
    team_id = request.args.get("team_id")
    if not team_id:
        return jsonify({"error":"Missing team_id"}), 400
    try:
        resp = requests.get(f"https://api-football-v1.p.rapidapi.com/v3/teams/statistics?team={team_id}&season=2023", headers=football_headers)
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/correct_score")
@limiter.limit("50/hour")
def correct_score():
    try:
        home_avg = float(request.args.get("home_avg",1.4))
        away_avg = float(request.args.get("away_avg",1.1))
    except:
        return jsonify({"error":"Invalid averages"}), 400
    grid = predictCorrectScore(home_avg, away_avg)
    top5 = dict(sorted(grid.items(), key=lambda x:-x[1])[:5])
    return jsonify({"probabilities": top5})

# --- Add other endpoints here as needed ---

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
