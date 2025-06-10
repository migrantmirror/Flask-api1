from flask import Flask, request, jsonify
import requests
import os
import math
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# API Keys
FOOTBALL_API_TOKEN = os.getenv("FOOTBALL_API_TOKEN")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Validate environment
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
odds_headers = {
    "x-api-key": ODDS_API_KEY
}

# --- Utility Functions ---

def poisson_prob(lmbda, k):
    """Poisson probability of scoring k goals given average lambda"""
    return (lmbda ** k * math.exp(-lmbda)) / math.factorial(k)

def kelly_criterion(prob, odds):
    """Calculates Kelly stake size"""
    if odds <= 1: return 0
    return max(0, (prob * (odds - 1) - (1 - prob)) / (odds - 1))

def implied_prob(odds):
    """Convert decimal odds to implied probability"""
    return 1 / odds if odds > 0 else 0

def validate_args(*args):
    """Ensure required query parameters exist"""
    for arg in args:
        if not request.args.get(arg):
            return jsonify({"error": f"Missing parameter: {arg}"}), 400
    return None

# --- API Endpoints ---

@app.route("/")
def home():
    return jsonify({"message": "Welcome to GoalStats API! Use /api/* endpoints for football data and predictions."})

@app.route("/api/fixtures")
def fixtures():
    error = validate_args("league_id")
    if error: return error
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures?league={request.args['league_id']}&season={request.args.get('season', '2023')}"
    return jsonify(requests.get(url, headers=football_headers).json())

@app.route("/api/standings")
def standings():
    error = validate_args("league_id")
    if error: return error
    url = f"https://api-football-v1.p.rapidapi.com/v3/standings?league={request.args['league_id']}&season={request.args.get('season', '2023')}"
    return jsonify(requests.get(url, headers=football_headers).json())

@app.route("/api/lineups")
def lineups():
    error = validate_args("match_id")
    if error: return error
    url = f"https://api-football-v1.p.rapidapi.com/v3/lineups?fixture={request.args['match_id']}"
    return jsonify(requests.get(url, headers=football_headers).json())

@app.route("/api/live_scores")
def live_scores():
    error = validate_args("league_id")
    if error: return error
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures?live=all&league={request.args['league_id']}"
    return jsonify(requests.get(url, headers=football_headers).json())

@app.route("/api/match_status")
def match_status():
    error = validate_args("match_id")
    if error: return error
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures?id={request.args['match_id']}"
    return jsonify(requests.get(url, headers=football_headers).json())

@app.route("/api/odds")
def odds():
    error = validate_args("match_id")
    if error: return error
    url = f"https://api-football-v1.p.rapidapi.com/v3/odds?fixture={request.args['match_id']}"
    return jsonify(requests.get(url, headers=football_headers).json())

@app.route("/api/history")
def history():
    error = validate_args("team_id")
    if error: return error
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures?team={request.args['team_id']}&last={request.args.get('last', 10)}"
    return jsonify(requests.get(url, headers=football_headers).json())

@app.route("/api/team_stats")
def team_stats():
    error = validate_args("team_id")
    if error: return error
    url = f"https://api-football-v1.p.rapidapi.com/v3/teams/statistics?team={request.args['team_id']}&season={request.args.get('season', '2023')}"
    return jsonify(requests.get(url, headers=football_headers).json())

@app.route("/api/injuries")
def injuries():
    error = validate_args("team_id")
    if error: return error
    url = f"https://api-football-v1.p.rapidapi.com/v3/injuries?team={request.args['team_id']}"
    return jsonify(requests.get(url, headers=football_headers).json())

@app.route("/api/weather")
def weather():
    city = request.args.get("city", "London")
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
    return jsonify(requests.get(url).json())

@app.route("/api/travel_distance")
def travel_distance():
    error = validate_args("home_city", "away_city")
    if error: return error
    # Placeholder
    return jsonify({
        "home_city": request.args["home_city"],
        "away_city": request.args["away_city"],
        "distance_km": 500  # static placeholder
    })

@app.route("/api/predict")
def predict():
    try:
        home_avg = float(request.args.get("home_avg", 1.4))
        away_avg = float(request.args.get("away_avg", 1.1))
        home_odds = float(request.args.get("home_odds", 2.1))
        draw_odds = float(request.args.get("draw_odds", 3.2))
        away_odds = float(request.args.get("away_odds", 3.3))

        max_goals = 5
        prob_home, prob_draw, prob_away = 0, 0, 0

        for h in range(max_goals):
            for a in range(max_goals):
                p = poisson_prob(home_avg, h) * poisson_prob(away_avg, a)
                if h > a: prob_home += p
                elif h == a: prob_draw += p
                else: prob_away += p

        probs = [prob_home, prob_draw, prob_away]
        odds = [home_odds, draw_odds, away_odds]
        outcomes = ["Home Win", "Draw", "Away Win"]

        value_bets = [outcomes[i] for i in range(3) if probs[i] * odds[i] > 1]
        stakes = [round(kelly_criterion(probs[i], odds[i]), 3) for i in range(3)]

        return jsonify({
            "prob_home_win": round(prob_home, 3),
            "prob_draw": round(prob_draw, 3),
            "prob_away_win": round(prob_away, 3),
            "value_bets": value_bets,
            "recommended_stakes": stakes
        })

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 400

@app.route("/api/backtest")
def backtest():
    return jsonify({"message": "Backtesting functionality to be implemented."})

# --- Run App ---

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
