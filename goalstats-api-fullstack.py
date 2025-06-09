from flask import Flask, request, jsonify
import requests
import os
import math
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load API keys from .env
FOOTBALL_API_TOKEN = os.getenv("FOOTBALL_API_TOKEN")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

if not FOOTBALL_API_TOKEN or not ODDS_API_KEY or not WEATHER_API_KEY:
    raise EnvironmentError("Missing required API keys: FOOTBALL_API_TOKEN, ODDS_API_KEY, WEATHER_API_KEY")

# API headers for API-Football (RapidAPI)
football_headers = {
    "X-RapidAPI-Key": FOOTBALL_API_TOKEN,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}

# OddsAPI headers
odds_headers = {
    "x-api-key": ODDS_API_KEY
}

# --- Helper Functions ---

def poisson_prob(lmbda, k):
    return (lmbda ** k * math.exp(-lmbda)) / math.factorial(k)

def kelly_criterion(prob, odds):
    return max(0, (prob * (odds - 1) - (1 - prob)) / (odds - 1))

def implied_prob(odds):
    # Convert decimal odds to implied probability
    if odds <= 0:
        return 0
    return 1 / odds

# --- Core API Endpoints ---

@app.route("/")
def home():
    return jsonify({"message": "Welcome to GoalStats API! Use /api/* endpoints for football data and predictions."})

@app.route("/api/fixtures")
def fixtures():
    league_id = request.args.get("league_id")
    season = request.args.get("season", "2023")
    if not league_id:
        return jsonify({"error": "Missing league_id"}), 400
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures?league={league_id}&season={season}"
    res = requests.get(url, headers=football_headers)
    return jsonify(res.json())

@app.route("/api/standings")
def standings():
    league_id = request.args.get("league_id")
    season = request.args.get("season", "2023")
    if not league_id:
        return jsonify({"error": "Missing league_id"}), 400
    url = f"https://api-football-v1.p.rapidapi.com/v3/standings?league={league_id}&season={season}"
    res = requests.get(url, headers=football_headers)
    return jsonify(res.json())

@app.route("/api/lineups")
def lineups():
    match_id = request.args.get("match_id")
    if not match_id:
        return jsonify({"error": "Missing match_id"}), 400
    url = f"https://api-football-v1.p.rapidapi.com/v3/lineups?fixture={match_id}"
    res = requests.get(url, headers=football_headers)
    return jsonify(res.json())

@app.route("/api/live_scores")
def live_scores():
    league_id = request.args.get("league_id")
    if not league_id:
        return jsonify({"error": "Missing league_id"}), 400
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures?live=all&league={league_id}"
    res = requests.get(url, headers=football_headers)
    return jsonify(res.json())

@app.route("/api/match_status")
def match_status():
    match_id = request.args.get("match_id")
    if not match_id:
        return jsonify({"error": "Missing match_id"}), 400
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures?id={match_id}"
    res = requests.get(url, headers=football_headers)
    return jsonify(res.json())

# --- Odds & Bookmaker Data ---

@app.route("/api/odds")
def odds():
    match_id = request.args.get("match_id")
    if not match_id:
        return jsonify({"error": "Missing match_id"}), 400

    # API-Football odds endpoint
    url = f"https://api-football-v1.p.rapidapi.com/v3/odds?fixture={match_id}"
    res = requests.get(url, headers=football_headers)
    odds_data = res.json()

    # Here you can enhance by fetching odds from multiple bookmakers (if you have access)
    # Or integrate OddsAPI for odds comparison & movement tracking

    return jsonify(odds_data)

# --- Historical Match Data ---

@app.route("/api/history")
def history():
    team_id = request.args.get("team_id")
    last_n = request.args.get("last", 10)
    if not team_id:
        return jsonify({"error": "Missing team_id"}), 400
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures?team={team_id}&last={last_n}"
    res = requests.get(url, headers=football_headers)
    return jsonify(res.json())

# --- Advanced Team Stats (Form, xG, Injuries) ---

@app.route("/api/team_stats")
def team_stats():
    team_id = request.args.get("team_id")
    season = request.args.get("season", "2023")
    if not team_id:
        return jsonify({"error": "Missing team_id"}), 400
    url = f"https://api-football-v1.p.rapidapi.com/v3/teams/statistics?team={team_id}&season={season}"
    res = requests.get(url, headers=football_headers)
    return jsonify(res.json())

@app.route("/api/injuries")
def injuries():
    team_id = request.args.get("team_id")
    if not team_id:
        return jsonify({"error": "Missing team_id"}), 400
    url = f"https://api-football-v1.p.rapidapi.com/v3/injuries?team={team_id}"
    res = requests.get(url, headers=football_headers)
    return jsonify(res.json())

# --- External Contextual Data ---

@app.route("/api/weather")
def weather():
    city = request.args.get("city", "London")
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
    res = requests.get(url)
    return jsonify(res.json())

# Example endpoint for travel distance (stub)
@app.route("/api/travel_distance")
def travel_distance():
    home_city = request.args.get("home_city")
    away_city = request.args.get("away_city")
    if not home_city or not away_city:
        return jsonify({"error": "Missing home_city or away_city"}), 400
    # Placeholder: Implement real distance API e.g. Google Maps Distance Matrix API here
    distance_km = 500  # example static value
    return jsonify({"home_city": home_city, "away_city": away_city, "distance_km": distance_km})

# --- Prediction & Betting Strategy ---

@app.route("/api/predict")
def predict():
    try:
        # Fetch parameters
        home_avg = float(request.args.get("home_avg", 1.4))
        away_avg = float(request.args.get("away_avg", 1.1))
        home_odds = float(request.args.get("home_odds", 2.1))
        draw_odds = float(request.args.get("draw_odds", 3.2))
        away_odds = float(request.args.get("away_odds", 3.3))

        max_goals = 5
        prob_home_win, prob_draw, prob_away_win = 0, 0, 0

        # Poisson model to estimate probabilities
        for h in range(max_goals):
            for a in range(max_goals):
                p = poisson_prob(home_avg, h) * poisson_prob(away_avg, a)
                if h > a:
                    prob_home_win += p
                elif h == a:
                    prob_draw += p
                else:
                    prob_away_win += p

        # Value bet detection
        prob_list = [prob_home_win, prob_draw, prob_away_win]
        odds_list = [home_odds, draw_odds, away_odds]

        value_bets = ["Home Win", "Draw", "Away Win"]
        value_flags = [prob * o > 1 for prob, o in zip(prob_list, odds_list)]

        # Suggested stake via Kelly Criterion
        stakes = [round(kelly_criterion(prob, odds), 3) for prob, odds in zip(prob_list, odds_list)]

        # Return results
        return jsonify({
            "prob_home_win": round(prob_home_win, 3),
            "prob_draw": round(prob_draw, 3),
            "prob_away_win": round(prob_away_win, 3),
            "value_bets": [v for v, f in zip(value_bets, value_flags) if f],
            "recommended_stakes": stakes
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- Backtesting Stub Endpoint ---

@app.route("/api/backtest")
def backtest():
    # You would implement historic data model testing here
    # For now, return static example response
    return jsonify({"message": "Backtesting functionality to be implemented."})

# --- Run the App ---

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
