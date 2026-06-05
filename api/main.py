from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.xgboost
import pandas as pd
import psycopg2
import numpy as np
import requests
import os

app = FastAPI(title="Octagon Analytics API", version="1.0.0")

# Config
MLFLOW_URI = os.environ["MLFLOW_URI"]
DB_CONFIG = {
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "host": os.environ["DB_HOST"],
    "port": os.environ.get("DB_PORT", "5432"),
}

mlflow.set_tracking_uri(MLFLOW_URI)

# Load production model at startup
def load_production_model():
    model_uri = "models:/ufc_fight_predictor@production"
    return mlflow.xgboost.load_model(model_uri)

model = load_production_model()

def get_db():
    return psycopg2.connect(**DB_CONFIG)

def get_fighter_stats(name: str) -> dict:
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                fighter_id, name, height, reach, stance,
                slpm, str_acc, sapm, str_def,
                td_avg, td_acc, td_def, sub_avg
            FROM fighters
            WHERE LOWER(name) = LOWER(%s)
            LIMIT 1
        """, (name,))
        row = cur.fetchone()
    conn.close()
    
    if not row:
        return None
    
    cols = ['fighter_id', 'name', 'height', 'reach', 'stance',
            'slpm', 'str_acc', 'sapm', 'str_def',
            'td_avg', 'td_acc', 'td_def', 'sub_avg']
    return dict(zip(cols, row))

def height_to_inches(val):
    try:
        parts = str(val).replace('"', '').split("'")
        return int(parts[0]) * 12 + int(parts[1].strip())
    except:
        return None

def pct_to_float(val):
    try:
        return float(str(val).replace('%', '')) / 100
    except:
        return None

def build_features(f1: dict, f2: dict) -> pd.DataFrame:
    def parse(f):
        def to_float(val):
            try:
                return float(val) if val is not None else None
            except:
                return None
    
        return {
            'height_in': height_to_inches(f.get('height')),
            'reach_in': float(str(f.get('reach', '')).replace('"', '').strip()) if f.get('reach') else None,
            'slpm': to_float(f.get('slpm')),
            'sapm': to_float(f.get('sapm')),
            'str_acc': pct_to_float(f.get('str_acc')),
            'str_def': pct_to_float(f.get('str_def')),
            'td_avg': to_float(f.get('td_avg')),
            'td_acc': pct_to_float(f.get('td_acc')),
            'td_def': pct_to_float(f.get('td_def')),
            'sub_avg': to_float(f.get('sub_avg')),
        }

    p1, p2 = parse(f1), parse(f2)
    
    features = {
        'height_diff':       (p1['height_in'] or 0) - (p2['height_in'] or 0),
        'reach_diff':        (p1['reach_in'] or 0)  - (p2['reach_in'] or 0),
        'slpm_diff':         (p1['slpm'] or 0)      - (p2['slpm'] or 0),
        'sapm_diff':         (p1['sapm'] or 0)      - (p2['sapm'] or 0),
        'str_acc_diff':      (p1['str_acc'] or 0)   - (p2['str_acc'] or 0),
        'str_def_diff':      (p1['str_def'] or 0)   - (p2['str_def'] or 0),
        'td_avg_diff':       (p1['td_avg'] or 0)    - (p2['td_avg'] or 0),
        'td_acc_diff':       (p1['td_acc'] or 0)    - (p2['td_acc'] or 0),
        'td_def_diff':       (p1['td_def'] or 0)    - (p2['td_def'] or 0),
        'sub_avg_diff':      (p1['sub_avg'] or 0)   - (p2['sub_avg'] or 0),
        'experience_diff':   0,
        'win_streak_diff':   0,
        'recent_form_diff':  0,
        'days_since_last_diff': 0,
        'age_diff':          0,
        'is_title_fight':    0,
        'stance_same':       1 if f1.get('stance') == f2.get('stance') else 0,
        'stance_ortho_vs_south': 1 if set([f1.get('stance'), f2.get('stance')]) == {'Orthodox', 'Southpaw'} else 0,
        'stance_other':      0,
        'stance_unknown':    0,
        'wc_Bantamweight':   0, 'wc_Catch Weight':     0,
        'wc_Featherweight':  0, 'wc_Flyweight':        0,
        'wc_Heavyweight':    0, 'wc_Light Heavyweight': 0,
        'wc_Lightweight':    0, 'wc_Middleweight':     0,
        'wc_Open Weight':    0, 'wc_Super Heavyweight': 0,
        'wc_Welterweight':   0, 'wc_Women\'s Bantamweight': 0,
        'wc_Women\'s Featherweight': 0, 'wc_Women\'s Flyweight': 0,
        'wc_Women\'s Strawweight': 0,
    }
    
    return pd.DataFrame([features])

def log_prediction(f1: dict, f2: dict, f1_prob: float, f2_prob: float, 
                   event_id: str = None, weight_class: str = None):
    import mlflow
    from mlflow.tracking import MlflowClient
    
    # Get current production model version and AUC
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient(MLFLOW_URI)
        prod = client.get_model_version_by_alias("ufc_fight_predictor", "production")
        prod_run = client.get_run(prod.run_id)
        model_version = int(prod.version)
        model_auc = prod_run.data.metrics.get("test_auc", None)
    except:
        model_version = None
        model_auc = None

    predicted_winner_id = f1['fighter_id'] if f1_prob > f2_prob else f2['fighter_id']

    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO predictions (
                event_id, fighter_1_id, fighter_2_id,
                fighter_1_name, fighter_2_name,
                fighter_1_win_prob, fighter_2_win_prob,
                predicted_winner_id, model_version, model_auc,
                weight_class, is_upcoming
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            event_id,
            f1['fighter_id'],
            f2['fighter_id'],
            f1['name'],
            f2['name'],
            f1_prob,
            f2_prob,
            predicted_winner_id,
            model_version,
            model_auc,
            weight_class,
            True
        ))
    conn.commit()
    conn.close()


# Request/Response models
class PredictionRequest(BaseModel):
    fighter_1: str
    fighter_2: str
    weight_class: str = None
    event_id: str = None

class PredictionResponse(BaseModel):
    fighter_1: str
    fighter_2: str
    fighter_1_win_probability: float
    fighter_2_win_probability: float
    predicted_winner: str
    confidence: str

@app.get("/")
def root():
    return {"message": "Octagon Analytics API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "ufc_fight_predictor@production"}

# I am going to make this function take in an optional fight id, log conditionall on that
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    f1 = get_fighter_stats(request.fighter_1)
    f2 = get_fighter_stats(request.fighter_2)
    
    if not f1:
        raise HTTPException(status_code=404, detail=f"Fighter not found: {request.fighter_1}")
    if not f2:
        raise HTTPException(status_code=404, detail=f"Fighter not found: {request.fighter_2}")
    
    # Set weight class if provided
    features = build_features(f1, f2)
    if request.weight_class and f"wc_{request.weight_class}" in features.columns:
        features[f"wc_{request.weight_class}"] = 1
    
    prob = model.predict_proba(features)[0]
    f1_prob = round(float(prob[1]), 4)
    f2_prob = round(float(prob[0]), 4)
    
    confidence = "high" if abs(f1_prob - 0.5) > 0.15 else "medium" if abs(f1_prob - 0.5) > 0.05 else "low"
    
    try:
        log_prediction(
            f1, f2, f1_prob, f2_prob,
            event_id=request.event_id if hasattr(request, 'event_id') else None,
            weight_class=request.weight_class
        )
    except Exception as e:
        print(f"Warning: failed to log prediction: {e}")

    return PredictionResponse(
        fighter_1=f1['name'],
        fighter_2=f2['name'],
        fighter_1_win_probability=f1_prob,
        fighter_2_win_probability=f2_prob,
        predicted_winner=f1['name'] if f1_prob > f2_prob else f2['name'],
        confidence=confidence
    )

@app.get("/fighter/{name}")
def get_fighter(name: str):
    fighter = get_fighter_stats(name)
    if not fighter:
        raise HTTPException(status_code=404, detail=f"Fighter not found: {name}")
    return fighter


@app.get("/predictions/history")
def get_prediction_history(limit: int = 100):
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT p.fighter_1_name, p.fighter_2_name,
                   p.fighter_1_win_prob, p.fighter_2_win_prob,
                   p.correct, p.weight_class, 
                   p.predicted_at, p.model_version,
                   e.name as event_name, e.date
            FROM predictions p
            LEFT JOIN events e ON p.event_id = e.event_id
            WHERE p.correct IS NOT NULL
            ORDER BY p.predicted_at DESC
            LIMIT %s
        """, (limit,))
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    conn.close()
    return [dict(zip(cols, row)) for row in rows]

@app.get("/predictions/upcoming")
def get_upcoming_predictions():
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT p.fighter_1_name, p.fighter_2_name,
                   p.fighter_1_win_prob, p.fighter_2_win_prob,
                   p.weight_class, p.model_version,
                   e.name as event_name, e.date
            FROM predictions p
            LEFT JOIN events e ON p.event_id = e.event_id
            WHERE p.is_upcoming = TRUE
            ORDER BY e.date ASC, p.predicted_at ASC
        """)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    conn.close()
    return [dict(zip(cols, row)) for row in rows]

@app.get("/predictions/accuracy/by-weight-class")
def get_accuracy_by_weight_class():
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT weight_class,
                   COUNT(*) as total,
                   SUM(CASE WHEN correct THEN 1 ELSE 0 END) as correct,
                   ROUND(AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END), 4) as accuracy
            FROM predictions
            WHERE correct IS NOT NULL
            AND weight_class IS NOT NULL
            GROUP BY weight_class
            ORDER BY total DESC
        """)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    conn.close()
    return [dict(zip(cols, row)) for row in rows]


@app.get("/predictions/accuracy")
def get_accuracy():
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                COUNT(*) as total_scored,
                SUM(CASE WHEN correct = TRUE THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN correct = FALSE THEN 1 ELSE 0 END) as incorrect,
                ROUND(AVG(CASE WHEN correct IS NOT NULL THEN 
                    CASE WHEN correct THEN 1.0 ELSE 0.0 END 
                END), 4) as accuracy
            FROM predictions
            WHERE correct IS NOT NULL
        """)
        row = cur.fetchone()
    conn.close()
    return {
        "total_scored": row[0],
        "correct": row[1],
        "incorrect": row[2],
        "accuracy": float(row[3]) if row[3] else None
    }

@app.get("/predictions/accuracy/by-version")
def get_accuracy_by_version():
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT model_version,
                   COUNT(*) as total,
                   SUM(CASE WHEN correct THEN 1 ELSE 0 END) as correct,
                   ROUND(AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END), 4) as accuracy,
                   model_auc
            FROM predictions
            WHERE correct IS NOT NULL
            GROUP BY model_version, model_auc
            ORDER BY model_version DESC
        """)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    conn.close()
    return [dict(zip(cols, row)) for row in rows]

@app.get("/model/current")
def get_current_model():
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient(MLFLOW_URI)
    try:
        prod = client.get_model_version_by_alias("ufc_fight_predictor", "production")
        run = client.get_run(prod.run_id)
        return {
            "version": prod.version,
            "auc": run.data.metrics.get("test_auc"),
            "accuracy": run.data.metrics.get("test_accuracy"),
            "log_loss": run.data.metrics.get("test_log_loss"),
            "created_at": prod.creation_timestamp,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fighters/search")
def search_fighters(q: str):
    conn = get_db()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT fighter_id, name, height, reach, stance, weight
            FROM fighters
            WHERE LOWER(name) LIKE LOWER(%s)
            ORDER BY name ASC
            LIMIT 10
        """, (f"%{q}%",))
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    conn.close()
    return [dict(zip(cols, row)) for row in rows]
