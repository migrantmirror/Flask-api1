from flask import Flask, request, jsonify
import requests
import os
import math
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load API keys
FOOTBALL_API_TOKEN = os.getenv("FOOTBALL_API_TOKEN")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Validate API keys on start
if not FOOTBALL_API_TOKEN or not WEATHER_API_KEY:
    raise EnvironmentError("Missing one or more required API keys in .env")

# API-Football headers
headers = {
    "X-RapidAPI-Key": FOOTBALL_API_TOKEN,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}

# --- Helper Functions ---
def poisson_prob(lmbda, k):
    return (lmbda**k * math.exp(-lmbda)) / math.factorial(k)

def kelly_criterion(prob, odds):
    return max(0, (prob * (odds - 1) - (1 - prob)) / (odds - 1))

# --- Endpoints ---

@app.route("/api/standings")
def standings():
    league_id = request.args.get("league_id")
    season = request.args.get("season", "2023")
    if not league_id:
        return jsonify({"error": "Missing league_id"}), 400
    url = f"https://api-football-v1.p.rapidapi.com/v3/standings?league={league_id}&season={season}"
    res = requests.get(url, headers=headers)
    return jsonify(res.json())

@app.route("/api/lineups")
def lineups():
    match_id = request.args.get("match_id")
    if not match_id:
        return jsonify({"error": "Missing match_id"}), 400
    url = f"https://api-football-v1.p.rapidapi.com/v3/lineups?fixture={match_id}"
    res = requests.get(url, headers=headers)
    return jsonify(res.json())

@app.route("/api/live")
def live():
    match_id = request.args.get("match_id")
    if not match_id:
        return jsonify({"error": "Missing match_id"}), 400
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures?id={match_id}"
    res = requests.get(url, headers=headers)
    return jsonify(res.json())

@app.route("/api/odds_detail")
def odds_detail():
    match_id = request.args.get("match_id")
    if not match_id:
        return jsonify({"error": "Missing match_id"}), 400
    url = f"https://api-football-v1.p.rapidapi.com/v3/odds?fixture={match_id}"
    res = requests.get(url, headers=headers)
    return jsonify(res.json())

@app.route("/api/history")
def history():
    team_id = request.args.get("team_id")
    if not team_id:
        return jsonify({"error": "Missing team_id"}), 400
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures?team={team_id}&last=10"
    res = requests.get(url, headers=headers)
    return jsonify(res.json())

@app.route("/api/weather")
def weather():
    city = request.args.get("city", "London")
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
    res = requests.get(url)
    return jsonify(res.json())

@app.route("/api/predict")
def predict():
    try:
        home_avg = float(request.args.get("home_avg", 1.4))
        away_avg = float(request.args.get("away_avg", 1.1))
        home_odds = float(request.args.get("home_odds", 2.1))
        draw_odds = float(request.args.get("draw_odds", 3.2))
        away_odds = float(request.args.get("away_odds", 3.3))

        max_goals = 5
        prob_home_win, prob_draw, prob_away_win = 0, 0, 0

        for h in range(max_goals):
            for a in range(max_goals):
                p = poisson_prob(home_avg, h) * poisson_prob(away_avg, a)
                if h > a:
                    prob_home_win += p
                elif h == a:
                    prob_draw += p
                else:
                    prob_away_win += p

        prob_list = [prob_home_win, prob_draw, prob_away_win]
        odds_list = [home_odds, draw_odds, away_odds]

        value_bets = ["Home", "Draw", "Away"]
        value_flags = [prob * o > 1 for prob, o in zip(prob_list, odds_list)]

        suggestions = [v for v, f in zip(value_bets, value_flags) if f]
        stakes = [round(kelly_criterion(prob, odds), 3) for prob, odds in zip(prob_list, odds_list)]

        return jsonify({
            "prob_home": round(prob_home_win, 2),
            "prob_draw": round(prob_draw, 2),
            "prob_away": round(prob_away_win, 2),
            "value_bets": suggestions,
            "suggested_stakes": stakes
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- Production Entry Point ---
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
