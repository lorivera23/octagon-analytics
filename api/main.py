import os

import mlflow
import mlflow.xgboost
import pandas as pd
import psycopg2
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel


app = FastAPI(title="Octagon Analytics API", version="1.0.0")

MLFLOW_URI = os.environ["MLFLOW_URI"]

WEIGHT_CLASS_FEATURES = [
    "wc_Bantamweight",
    "wc_Catch Weight",
    "wc_Featherweight",
    "wc_Flyweight",
    "wc_Heavyweight",
    "wc_Light Heavyweight",
    "wc_Lightweight",
    "wc_Middleweight",
    "wc_Open Weight",
    "wc_Super Heavyweight",
    "wc_Welterweight",
    "wc_Women's Bantamweight",
    "wc_Women's Featherweight",
    "wc_Women's Flyweight",
    "wc_Women's Strawweight",
]

FEATURE_COLS = [
    "height_diff",
    "reach_diff",
    "experience_diff",
    "win_streak_diff",
    "recent_form_diff",
    "days_since_last_diff",
    "age_diff",
    "stance_same",
    "stance_ortho_vs_south",
    "stance_other",
    "stance_unknown",
    *WEIGHT_CLASS_FEATURES,
]


def get_db():
    return psycopg2.connect(
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        host=os.environ["DB_HOST"],
        port=os.environ.get("DB_PORT", "5432"),
    )


mlflow.set_tracking_uri(MLFLOW_URI)


def load_production_model():
    client = MlflowClient(MLFLOW_URI)

    production = client.get_model_version_by_alias(
        "ufc_fight_predictor",
        "production",
    )

    model = mlflow.xgboost.load_model(
        f"models:/ufc_fight_predictor/{production.version}"
    )

    auc = client.get_run(
        production.run_id
    ).data.metrics.get("test_auc")

    return model, int(production.version), auc


model, MODEL_VERSION, MODEL_AUC = load_production_model()


class FighterNotFoundError(Exception):
    def __init__(self, name):
        self.name = name
        super().__init__(f"Fighter not found: {name}")


def get_fighter_stats(name: str) -> dict | None:
    """
    Return the stored fighter profile.

    Career aggregate striking/grappling statistics remain available to the
    frontend, but they are intentionally not used by the prediction model
    because they represent a current snapshot rather than historical values.
    """
    conn = get_db()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    fighter_id,
                    name,
                    height,
                    reach,
                    stance,
                    dob,
                    weight,
                    slpm,
                    str_acc,
                    sapm,
                    str_def,
                    td_avg,
                    td_acc,
                    td_def,
                    sub_avg
                FROM fighters
                WHERE LOWER(name) = LOWER(%s)
                LIMIT 1
                """,
                (name,),
            )

            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return None

    columns = [
        "fighter_id",
        "name",
        "height",
        "reach",
        "stance",
        "dob",
        "weight",
        "slpm",
        "str_acc",
        "sapm",
        "str_def",
        "td_avg",
        "td_acc",
        "td_def",
        "sub_avg",
    ]

    return dict(zip(columns, row))


def height_to_inches(value):
    try:
        feet, inches = str(value).replace('"', "").split("'")
        return int(feet) * 12 + int(inches.strip())
    except (TypeError, ValueError):
        return None


def reach_to_float(value):
    try:
        return float(str(value).replace('"', "").strip())
    except (TypeError, ValueError):
        return None


def numeric_difference(value_1, value_2):
    if value_1 is None or value_2 is None:
        return None

    return value_1 - value_2


def get_prediction_date(conn, event_id: str | None):
    if event_id:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT date
                FROM events
                WHERE event_id = %s
                """,
                (event_id,),
            )

            row = cur.fetchone()

        if row and row[0]:
            return pd.Timestamp(row[0])

    return pd.Timestamp.today().normalize()


def get_fighter_history(
    conn,
    fighter_id: str,
    dob,
    as_of_date,
):
    """
    Calculate fighter state using only fights that occurred before the
    requested prediction date.
    """
    as_of_date = pd.Timestamp(as_of_date)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                e.date,
                fi.winner_id
            FROM fights fi
            JOIN events e
                ON fi.event_id = e.event_id
            WHERE
                (
                    fi.fighter_1_id = %s
                    OR fi.fighter_2_id = %s
                )
                AND fi.winner_id IS NOT NULL
                AND e.date < %s
            ORDER BY e.date ASC, fi.fight_id ASC
            """,
            (
                fighter_id,
                fighter_id,
                as_of_date.date(),
            ),
        )

        rows = cur.fetchall()

    wins = [
        winner_id == fighter_id
        for _, winner_id in rows
    ]

    win_streak = 0

    for won in reversed(wins):
        if not won:
            break

        win_streak += 1

    recent_results = wins[-5:]

    if recent_results:
        recent_form = sum(recent_results) / len(recent_results)
    else:
        recent_form = 0.5

    if rows:
        last_fight_date = pd.Timestamp(rows[-1][0])
        days_since_last = (as_of_date - last_fight_date).days
    else:
        days_since_last = None

    dob = pd.to_datetime(dob, errors="coerce")

    age = (
        (as_of_date - dob).days / 365.25
        if pd.notna(dob)
        else None
    )

    return {
        "experience": len(rows),
        "win_streak": win_streak,
        "recent_form": recent_form,
        "days_since_last": days_since_last,
        "age": age,
    }


def build_features(
    conn,
    fighter_1: dict,
    fighter_2: dict,
    weight_class: str | None = None,
    event_id: str | None = None,
) -> pd.DataFrame:
    prediction_date = get_prediction_date(
        conn,
        event_id,
    )

    fighter_1_history = get_fighter_history(
        conn,
        fighter_1["fighter_id"],
        fighter_1.get("dob"),
        prediction_date,
    )

    fighter_2_history = get_fighter_history(
        conn,
        fighter_2["fighter_id"],
        fighter_2.get("dob"),
        prediction_date,
    )

    fighter_1_height = height_to_inches(
        fighter_1.get("height")
    )
    fighter_2_height = height_to_inches(
        fighter_2.get("height")
    )

    fighter_1_reach = reach_to_float(
        fighter_1.get("reach")
    )
    fighter_2_reach = reach_to_float(
        fighter_2.get("reach")
    )

    stance_1 = (
        fighter_1.get("stance", "").strip().lower()
        if fighter_1.get("stance")
        else None
    )

    stance_2 = (
        fighter_2.get("stance", "").strip().lower()
        if fighter_2.get("stance")
        else None
    )

    if not stance_1 or not stance_2:
        stance_matchup = "unknown"
    elif stance_1 == stance_2:
        stance_matchup = "same"
    elif {stance_1, stance_2} == {"orthodox", "southpaw"}:
        stance_matchup = "ortho_vs_south"
    else:
        stance_matchup = "other"

    features = {
        "height_diff": numeric_difference(
            fighter_1_height,
            fighter_2_height,
        ),
        "reach_diff": numeric_difference(
            fighter_1_reach,
            fighter_2_reach,
        ),
        "experience_diff": (
            fighter_1_history["experience"]
            - fighter_2_history["experience"]
        ),
        "win_streak_diff": (
            fighter_1_history["win_streak"]
            - fighter_2_history["win_streak"]
        ),
        "recent_form_diff": (
            fighter_1_history["recent_form"]
            - fighter_2_history["recent_form"]
        ),
        "days_since_last_diff": numeric_difference(
            fighter_1_history["days_since_last"],
            fighter_2_history["days_since_last"],
        ),
        "age_diff": numeric_difference(
            fighter_1_history["age"],
            fighter_2_history["age"],
        ),
        "stance_same": int(stance_matchup == "same"),
        "stance_ortho_vs_south": int(
            stance_matchup == "ortho_vs_south"
        ),
        "stance_other": int(stance_matchup == "other"),
        "stance_unknown": int(
            stance_matchup == "unknown"
        ),
    }

    for feature in WEIGHT_CLASS_FEATURES:
        features[feature] = 0

    weight_feature = (
        f"wc_{weight_class}"
        if weight_class
        else None
    )

    if weight_feature in WEIGHT_CLASS_FEATURES:
        features[weight_feature] = 1

    feature_frame = pd.DataFrame(
        [features],
        columns=FEATURE_COLS,
    )

    # Training uses a constant-zero imputer, so inference mirrors the same
    # preprocessing for unavailable measurements.
    return feature_frame.fillna(0)


def store_prediction(
    fighter_1: dict,
    fighter_2: dict,
    fighter_1_probability: float,
    fighter_2_probability: float,
    event_id: str,
    fight_id: str,
    weight_class: str | None,
):
    predicted_winner_id = (
        fighter_1["fighter_id"]
        if fighter_1_probability > fighter_2_probability
        else fighter_2["fighter_id"]
    )

    conn = get_db()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions (
                    fight_id,
                    event_id,
                    fighter_1_id,
                    fighter_2_id,
                    fighter_1_name,
                    fighter_2_name,
                    fighter_1_win_prob,
                    fighter_2_win_prob,
                    predicted_winner_id,
                    model_version,
                    model_auc,
                    weight_class,
                    is_upcoming
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, TRUE
                )
                """,
                (
                    fight_id,
                    event_id,
                    fighter_1["fighter_id"],
                    fighter_2["fighter_id"],
                    fighter_1["name"],
                    fighter_2["name"],
                    fighter_1_probability,
                    fighter_2_probability,
                    predicted_winner_id,
                    MODEL_VERSION,
                    MODEL_AUC,
                    weight_class,
                ),
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def predict_fight(
    fighter_1_name: str,
    fighter_2_name: str,
    weight_class: str | None = None,
    event_id: str | None = None,
):
    fighter_1 = get_fighter_stats(
        fighter_1_name
    )
    fighter_2 = get_fighter_stats(
        fighter_2_name
    )

    if not fighter_1:
        raise FighterNotFoundError(
            fighter_1_name
        )

    if not fighter_2:
        raise FighterNotFoundError(
            fighter_2_name
        )

    conn = get_db()

    try:
        features = build_features(
            conn,
            fighter_1,
            fighter_2,
            weight_class=weight_class,
            event_id=event_id,
        )
    finally:
        conn.close()

    probabilities = model.predict_proba(
        features
    )[0]

    fighter_1_probability = round(
        float(probabilities[1]),
        4,
    )

    fighter_2_probability = round(
        float(probabilities[0]),
        4,
    )

    margin = abs(
        fighter_1_probability - 0.5
    )

    if margin > 0.15:
        confidence = "high"
    elif margin > 0.05:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "fighter_1": fighter_1["name"],
        "fighter_2": fighter_2["name"],
        "fighter_1_win_probability": fighter_1_probability,
        "fighter_2_win_probability": fighter_2_probability,
        "predicted_winner": (
            fighter_1["name"]
            if fighter_1_probability
            > fighter_2_probability
            else fighter_2["name"]
        ),
        "confidence": confidence,
        "model_version": MODEL_VERSION,
        "model_auc": MODEL_AUC,
        "_fighter_1": fighter_1,
        "_fighter_2": fighter_2,
    }


class PredictionRequest(BaseModel):
    fighter_1: str
    fighter_2: str
    weight_class: str | None = None
    event_id: str | None = None
    fight_id: str | None = None


class PredictionResponse(BaseModel):
    fighter_1: str
    fighter_2: str
    fighter_1_win_probability: float
    fighter_2_win_probability: float
    predicted_winner: str
    confidence: str
    model_version: int
    model_auc: float | None

@app.get("/")
def root():
    return {"message": "Octagon Analytics API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "ufc_fight_predictor@production"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        result = predict_fight(
            request.fighter_1,
            request.fighter_2,
            weight_class=request.weight_class,
            event_id=request.event_id,
        )
    except FighterNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    fighter_1 = result.pop("_fighter_1")
    fighter_2 = result.pop("_fighter_2")

    # Upcoming-fight requests include these identifiers. Custom matchup
    # requests from the frontend do not need to be persisted.
    if request.fight_id and request.event_id:
        store_prediction(
            fighter_1,
            fighter_2,
            result["fighter_1_win_probability"],
            result["fighter_2_win_probability"],
            event_id=request.event_id,
            fight_id=request.fight_id,
            weight_class=request.weight_class,
        )

    return PredictionResponse(**result)

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

# change to get veresions from load_production model 
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
