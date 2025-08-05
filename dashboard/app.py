from flask import Flask, jsonify, render_template
from pathlib import Path
import pandas as pd
import json

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parents[1] / "templates"),
    static_folder=str(Path(__file__).resolve().parents[1] / "static"),
)

BASE = Path(__file__).resolve().parents[2]  # project_root
DATA = BASE / "data"
REPORTS = BASE / "reports"

def load_prices():
    fp = DATA / "cleaned_oil_prices.csv"
    if not fp.exists():
        return []
    df = pd.read_csv(fp)
    # Ensure expected columns
    if "Date" not in df.columns or "Price" not in df.columns:
        return []
    # Keep it light: stringify date for JS
    return df[["Date", "Price"]].to_dict(orient="records")

def load_json(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None

@app.route("/")
def index():
    return render_template("index.html")

@app.get("/api/price")
def api_price():
    return jsonify(load_prices())

@app.get("/api/changepoint")
def api_changepoint():
    return jsonify(load_json(REPORTS / "task2_summary.json") or {})

@app.get("/api/windowstats")
def api_windowstats():
    return jsonify(load_json(REPORTS / "task2_price_window_stats.json") or {})

@app.get("/api/events_near")
def api_events_near():
    fp = REPORTS / "task2_nearby_events.csv"
    if fp.exists():
        df = pd.read_csv(fp)
        return jsonify(df.to_dict(orient="records"))
    return jsonify([])

if __name__ == "__main__":
    # Run: python app/backend/app.py
    app.run(host="0.0.0.0", port=5050, debug=True)
