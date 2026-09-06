import os
from collections import defaultdict

import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow.models import infer_signature
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sqlalchemy import create_engine
from xgboost import XGBClassifier


DB_URI = (
    f"postgresql://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
    f"@{os.environ['DB_HOST']}:{os.environ.get('DB_PORT', '5432')}"
    f"/{os.environ['DB_NAME']}"
)

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

# These features reverse sign when fighter 1 and fighter 2 are swapped.
DIRECTIONAL_FEATURES = [
    "height_diff",
    "reach_diff",
    "experience_diff",
    "win_streak_diff",
    "recent_form_diff",
    "days_since_last_diff",
    "age_diff",
]


def get_engine():
    return create_engine(DB_URI)


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


def compute_fighter_history(history_df):
    """
    Build pre-fight history features in chronological order.

    Fighter state is calculated before the current fight result is added,
    preventing the current fight from leaking into its own features.
    """
    fighter_record = defaultdict(list)
    results = []

    for _, row in history_df.iterrows():
        fight_date = row["date"]

        def get_stats(fighter_id, dob):
            past_fights = fighter_record[fighter_id]
            wins = [fight["won"] for fight in past_fights]

            if not past_fights:
                return {
                    "experience": 0,
                    "win_streak": 0,
                    "recent_form": 0.5,
                    "days_since_last": None,
                    "age": (
                        (fight_date - dob).days / 365.25
                        if pd.notna(dob)
                        else None
                    ),
                }

            win_streak = 0
            for won in reversed(wins):
                if not won:
                    break
                win_streak += 1

            recent_results = wins[-5:]

            return {
                "experience": len(past_fights),
                "win_streak": win_streak,
                "recent_form": sum(recent_results) / len(recent_results),
                "days_since_last": (
                    fight_date - past_fights[-1]["date"]
                ).days,
                "age": (
                    (fight_date - dob).days / 365.25
                    if pd.notna(dob)
                    else None
                ),
            }

        fighter_1_stats = get_stats(
            row["fighter_1_id"],
            row["fighter_1_dob"],
        )
        fighter_2_stats = get_stats(
            row["fighter_2_id"],
            row["fighter_2_dob"],
        )

        results.append(
            {
                "fight_id": row["fight_id"],
                "f1_experience": fighter_1_stats["experience"],
                "f1_win_streak": fighter_1_stats["win_streak"],
                "f1_recent_form": fighter_1_stats["recent_form"],
                "f1_days_since_last": fighter_1_stats["days_since_last"],
                "f1_age": fighter_1_stats["age"],
                "f2_experience": fighter_2_stats["experience"],
                "f2_win_streak": fighter_2_stats["win_streak"],
                "f2_recent_form": fighter_2_stats["recent_form"],
                "f2_days_since_last": fighter_2_stats["days_since_last"],
                "f2_age": fighter_2_stats["age"],
            }
        )

        fighter_1_won = row["winner_id"] == row["fighter_1_id"]
        fighter_2_won = row["winner_id"] == row["fighter_2_id"]

        fighter_record[row["fighter_1_id"]].append(
            {"date": fight_date, "won": fighter_1_won}
        )
        fighter_record[row["fighter_2_id"]].append(
            {"date": fight_date, "won": fighter_2_won}
        )

    return pd.DataFrame(results)


def build_dataset(engine):
    query = """
    SELECT
        fi.fight_id,
        fi.fighter_1_id,
        fi.fighter_2_id,
        fi.winner_id,
        e.date,
        fi.weight_class,
        f1.height AS f1_height,
        f1.reach AS f1_reach,
        f1.stance AS f1_stance,
        f2.height AS f2_height,
        f2.reach AS f2_reach,
        f2.stance AS f2_stance,
        CASE
            WHEN fi.winner_id = fi.fighter_1_id THEN 1
            ELSE 0
        END AS fighter_1_won
    FROM fights fi
    JOIN events e
        ON fi.event_id = e.event_id
    JOIN fighters f1
        ON fi.fighter_1_id = f1.fighter_id
    JOIN fighters f2
        ON fi.fighter_2_id = f2.fighter_id
    WHERE fi.winner_id IS NOT NULL
    """

    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])

    history_query = """
    SELECT
        fi.fight_id,
        fi.fighter_1_id,
        fi.fighter_2_id,
        fi.winner_id,
        e.date,
        f1.dob AS fighter_1_dob,
        f2.dob AS fighter_2_dob
    FROM fights fi
    JOIN events e
        ON fi.event_id = e.event_id
    JOIN fighters f1
        ON fi.fighter_1_id = f1.fighter_id
    JOIN fighters f2
        ON fi.fighter_2_id = f2.fighter_id
    WHERE fi.winner_id IS NOT NULL
    ORDER BY e.date ASC, fi.fight_id ASC
    """

    history = pd.read_sql(history_query, engine)
    history["date"] = pd.to_datetime(history["date"])
    history["fighter_1_dob"] = pd.to_datetime(
        history["fighter_1_dob"], errors="coerce"
    )
    history["fighter_2_dob"] = pd.to_datetime(
        history["fighter_2_dob"], errors="coerce"
    )

    fighter_features = compute_fighter_history(history)

    df = df.merge(
        fighter_features,
        on="fight_id",
        how="inner",
    )

    df["f1_height_in"] = df["f1_height"].apply(height_to_inches)
    df["f2_height_in"] = df["f2_height"].apply(height_to_inches)
    df["f1_reach_in"] = df["f1_reach"].apply(reach_to_float)
    df["f2_reach_in"] = df["f2_reach"].apply(reach_to_float)

    df["height_diff"] = df["f1_height_in"] - df["f2_height_in"]
    df["reach_diff"] = df["f1_reach_in"] - df["f2_reach_in"]
    df["experience_diff"] = df["f1_experience"] - df["f2_experience"]
    df["win_streak_diff"] = df["f1_win_streak"] - df["f2_win_streak"]
    df["recent_form_diff"] = (
        df["f1_recent_form"] - df["f2_recent_form"]
    )
    df["days_since_last_diff"] = (
        df["f1_days_since_last"] - df["f2_days_since_last"]
    )
    df["age_diff"] = df["f1_age"] - df["f2_age"]

    def stance_matchup(stance_1, stance_2):
        if pd.isna(stance_1) or pd.isna(stance_2):
            return "unknown"

        stance_1 = stance_1.strip().lower()
        stance_2 = stance_2.strip().lower()

        if stance_1 == stance_2:
            return "same"

        if {stance_1, stance_2} == {"orthodox", "southpaw"}:
            return "ortho_vs_south"

        return "other"

    df["stance_matchup"] = df.apply(
        lambda row: stance_matchup(
            row["f1_stance"],
            row["f2_stance"],
        ),
        axis=1,
    )

    stance_dummies = pd.get_dummies(
        df["stance_matchup"],
        prefix="stance",
        dtype=int,
    )
    weight_dummies = pd.get_dummies(
        df["weight_class"],
        prefix="wc",
        dtype=int,
    )

    df = pd.concat(
        [df, stance_dummies, weight_dummies],
        axis=1,
    )

    # Keep the feature schema stable even when a category does not appear
    # in a particular scrape.
    for feature in FEATURE_COLS:
        if feature not in df.columns:
            df[feature] = 0

    return df.sort_values(["date", "fight_id"]).reset_index(drop=True)


def chronological_split(df, test_fraction=0.20):
    """
    Reserve the newest portion of event dates for evaluation.

    This more closely reflects the real task: train on past UFC fights and
    predict fights that occur later.
    """
    unique_dates = sorted(df["date"].dropna().unique())

    if len(unique_dates) < 2:
        raise ValueError("Not enough event dates for a chronological split.")

    split_index = max(
        1,
        min(
            len(unique_dates) - 1,
            int(len(unique_dates) * (1 - test_fraction)),
        ),
    )

    cutoff_date = unique_dates[split_index]

    train_df = df[df["date"] < cutoff_date].copy()
    test_df = df[df["date"] >= cutoff_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Chronological split produced an empty training or test set."
        )

    return train_df, test_df, pd.Timestamp(cutoff_date)


def mirror_training_data(X, y):
    """
    Add the opposite fighter ordering for training only.

    A fight A-vs-B becomes B-vs-A, directional differences reverse sign,
    and the target is inverted. The test set is never duplicated.
    """
    mirrored_X = X.copy()

    for feature in DIRECTIONAL_FEATURES:
        mirrored_X[feature] = -mirrored_X[feature]

    mirrored_y = 1 - y

    augmented_X = pd.concat(
        [X, mirrored_X],
        ignore_index=True,
    )
    augmented_y = pd.concat(
        [
            y.reset_index(drop=True),
            mirrored_y.reset_index(drop=True),
        ],
        ignore_index=True,
    )

    return augmented_X, augmented_y


def train():
    engine = get_engine()

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("ufc_fight_prediction")

    print("Building dataset...")
    df = build_dataset(engine)

    train_df, test_df, cutoff_date = chronological_split(df)

    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df["fighter_1_won"].astype(int)

    X_test = test_df[FEATURE_COLS].copy()
    y_test = test_df["fighter_1_won"].astype(int)

    X_train, y_train = mirror_training_data(X_train, y_train)

    # Fit preprocessing only on training data.
    imputer = SimpleImputer(strategy="constant", fill_value=0)

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=FEATURE_COLS,
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=FEATURE_COLS,
    )

    params = {
        "n_estimators": 700,
        "max_depth": 4,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
    }

    with mlflow.start_run(run_name="xgboost_weekly_retrain") as run:
        mlflow.log_params(
            {
                "model_type": "XGBoost",
                **params,
                "split_strategy": "chronological",
                "test_cutoff_date": cutoff_date.date().isoformat(),
                "training_augmentation": "fighter_order_mirroring",
            }
        )

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        train_probs = model.predict_proba(X_train)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]

        train_predictions = (train_probs >= 0.5).astype(int)
        test_predictions = (test_probs >= 0.5).astype(int)

        train_accuracy = accuracy_score(
            y_train,
            train_predictions,
        )
        test_accuracy = accuracy_score(
            y_test,
            test_predictions,
        )
        test_auc = roc_auc_score(
            y_test,
            test_probs,
        )
        test_log_loss = log_loss(
            y_test,
            test_probs,
        )

        mlflow.log_metrics(
            {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "test_auc": test_auc,
                "test_log_loss": test_log_loss,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            }
        )

        signature = infer_signature(
            X_train,
            model.predict_proba(X_train),
        )

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
        )

        model_uri = f"runs:/{run.info.run_id}/model"

        registered = mlflow.register_model(
            model_uri,
            "ufc_fight_predictor",
        )

        new_version = registered.version

        print(f"Train rows:      {len(X_train)}")
        print(f"Test rows:       {len(X_test)}")
        print(f"Test cutoff:     {cutoff_date.date()}")
        print(f"Train Accuracy:  {train_accuracy:.4f}")
        print(f"Test Accuracy:   {test_accuracy:.4f}")
        print(f"Test AUC:        {test_auc:.4f}")
        print(f"Test Log Loss:   {test_log_loss:.4f}")
        print(f"new_version:{new_version}")


if __name__ == "__main__":
    train()