import sys
import os
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from collections import defaultdict
import os

DB_URI = f"postgresql://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}@{os.environ['DB_HOST']}:{os.environ.get('DB_PORT','5432')}/{os.environ['DB_NAME']}"
MLFLOW_URI = os.environ["MLFLOW_URI"]


def get_engine():
    return create_engine(DB_URI)

def pct_to_float(val):
    try:
        return float(str(val).replace('%', '')) / 100
    except:
        return None

def height_to_inches(val):
    try:
        parts = str(val).replace('"', '').split("'")
        return int(parts[0]) * 12 + int(parts[1].strip())
    except:
        return None

def reach_to_float(val):
    try:
        return float(str(val).replace('"', '').strip())
    except:
        return None

def clean_method(method):
    if pd.isna(method):
        return 'Unknown'
    method = method.upper()
    if 'DEC' in method:
        return 'Decision'
    elif 'KO' in method or 'TKO' in method:
        return 'KO/TKO'
    elif any(x in method for x in ['CHOKE','LOCK','BAR','TRIANGLE','ANACONDA','GUILLOTINE','SUB']):
        return 'Submission'
    elif 'DRAW' in method or 'NC' in method:
        return 'Draw/NC'
    else:
        return 'Other'

def compute_fighter_history(history_df):
    fighter_record = defaultdict(list)
    results = []

    for _, row in history_df.iterrows():
        fight_id = row['fight_id']
        winner_id = row['winner_id']
        loser_id = row['loser_id']
        date = row['date']

        def get_stats(fighter_id, dob):
            past_fights = fighter_record[fighter_id]
            n = len(past_fights)
            if n == 0:
                return {
                    'experience': 0, 'win_streak': 0, 'recent_form': 0.5,
                    'days_since_last': None,
                    'age': (date - dob).days / 365.25 if pd.notna(dob) else None
                }
            wins = [f['won'] for f in past_fights]
            dates = [f['date'] for f in past_fights]
            streak = 0
            for w in reversed(wins):
                if w: streak += 1
                else: break
            recent = wins[-5:]
            return {
                'experience': n,
                'win_streak': streak,
                'recent_form': sum(recent) / len(recent),
                'days_since_last': (date - dates[-1]).days,
                'age': (date - dob).days / 365.25 if pd.notna(dob) else None
            }

        winner_stats = get_stats(winner_id, row['winner_dob'])
        loser_stats = get_stats(loser_id, row['loser_dob'])

        results.append({
            'fight_id': fight_id,
            'f1_experience': winner_stats['experience'],
            'f1_win_streak': winner_stats['win_streak'],
            'f1_recent_form': winner_stats['recent_form'],
            'f1_days_since_last': winner_stats['days_since_last'],
            'f1_age': winner_stats['age'],
            'f2_experience': loser_stats['experience'],
            'f2_win_streak': loser_stats['win_streak'],
            'f2_recent_form': loser_stats['recent_form'],
            'f2_days_since_last': loser_stats['days_since_last'],
            'f2_age': loser_stats['age'],
        })

        fighter_record[winner_id].append({'date': date, 'won': True})
        fighter_record[loser_id].append({'date': date, 'won': False})

    return pd.DataFrame(results)

def build_dataset(engine):
    # Load joined dataset
    query = """
    SELECT fi.fight_id, fi.method, e.date, fi.weight_class, fi.is_title_fight,
        f1.name as fighter_1_name, f1.height as f1_height, f1.reach as f1_reach,
        f1.stance as f1_stance, f1.slpm as f1_slpm, f1.str_acc as f1_str_acc,
        f1.sapm as f1_sapm, f1.str_def as f1_str_def, f1.td_avg as f1_td_avg,
        f1.td_acc as f1_td_acc, f1.td_def as f1_td_def, f1.sub_avg as f1_sub_avg,
        f2.name as fighter_2_name, f2.height as f2_height, f2.reach as f2_reach,
        f2.stance as f2_stance, f2.slpm as f2_slpm, f2.str_acc as f2_str_acc,
        f2.sapm as f2_sapm, f2.str_def as f2_str_def, f2.td_avg as f2_td_avg,
        f2.td_acc as f2_td_acc, f2.td_def as f2_td_def, f2.sub_avg as f2_sub_avg,
        CASE WHEN fi.winner_id = fi.fighter_1_id THEN 1 ELSE 0 END as fighter_1_won
    FROM fights fi
    JOIN events e ON fi.event_id = e.event_id
    JOIN fighters f1 ON fi.fighter_1_id = f1.fighter_id
    JOIN fighters f2 ON fi.fighter_2_id = f2.fighter_id
    WHERE fi.winner_id IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    df['method_clean'] = df['method'].apply(clean_method)

    # Fighter history
    history_query = """
    SELECT fi.fight_id, fi.fighter_1_id as winner_id, fi.fighter_2_id as loser_id,
        e.date, f1.dob as winner_dob, f2.dob as loser_dob
    FROM fights fi
    JOIN events e ON fi.event_id = e.event_id
    JOIN fighters f1 ON fi.fighter_1_id = f1.fighter_id
    JOIN fighters f2 ON fi.fighter_2_id = f2.fighter_id
    WHERE fi.winner_id IS NOT NULL
    ORDER BY e.date ASC
    """
    history = pd.read_sql(history_query, engine)
    history['date'] = pd.to_datetime(history['date'])
    history['winner_dob'] = pd.to_datetime(history['winner_dob'])
    history['loser_dob'] = pd.to_datetime(history['loser_dob'])

    fighter_features = compute_fighter_history(history)
    df = df.merge(fighter_features, on='fight_id', how='left')

    # Flip dataset
    df_flipped = df.copy()
    f1_cols = [c for c in df.columns if c.startswith('f1_') or c == 'fighter_1_name']
    f2_cols = [c for c in df.columns if c.startswith('f2_') or c == 'fighter_2_name']
    rename_map = {}
    for c in f1_cols:
        rename_map[c] = c.replace('f1_', 'f2_').replace('fighter_1_', 'fighter_2_')
    for c in f2_cols:
        rename_map[c] = c.replace('f2_', 'f1_').replace('fighter_2_', 'fighter_1_')
    df_flipped = df_flipped.rename(columns=rename_map)
    df_flipped['fighter_1_won'] = 0

    flipped_mask = df_flipped['fighter_1_won'] == 0
    hist_f1_cols = ['f1_experience','f1_win_streak','f1_recent_form','f1_days_since_last','f1_age']
    hist_f2_cols = ['f2_experience','f2_win_streak','f2_recent_form','f2_days_since_last','f2_age']
    for f1_col, f2_col in zip(hist_f1_cols, hist_f2_cols):
        df_flipped.loc[flipped_mask, [f1_col, f2_col]] = \
            df_flipped.loc[flipped_mask, [f2_col, f1_col]].values

    df_balanced = pd.concat([df, df_flipped], ignore_index=True)

    # Conversions
    for prefix in ['f1', 'f2']:
        df_balanced[f'{prefix}_height_in'] = df_balanced[f'{prefix}_height'].apply(height_to_inches)
        df_balanced[f'{prefix}_reach_in'] = df_balanced[f'{prefix}_reach'].apply(reach_to_float)
        df_balanced[f'{prefix}_str_acc_f'] = df_balanced[f'{prefix}_str_acc'].apply(pct_to_float)
        df_balanced[f'{prefix}_str_def_f'] = df_balanced[f'{prefix}_str_def'].apply(pct_to_float)
        df_balanced[f'{prefix}_td_acc_f'] = df_balanced[f'{prefix}_td_acc'].apply(pct_to_float)
        df_balanced[f'{prefix}_td_def_f'] = df_balanced[f'{prefix}_td_def'].apply(pct_to_float)

    # Differentials
    df_balanced['height_diff'] = df_balanced['f1_height_in'] - df_balanced['f2_height_in']
    df_balanced['reach_diff'] = df_balanced['f1_reach_in'] - df_balanced['f2_reach_in']
    df_balanced['slpm_diff'] = df_balanced['f1_slpm'] - df_balanced['f2_slpm']
    df_balanced['sapm_diff'] = df_balanced['f1_sapm'] - df_balanced['f2_sapm']
    df_balanced['str_acc_diff'] = df_balanced['f1_str_acc_f'] - df_balanced['f2_str_acc_f']
    df_balanced['str_def_diff'] = df_balanced['f1_str_def_f'] - df_balanced['f2_str_def_f']
    df_balanced['td_avg_diff'] = df_balanced['f1_td_avg'] - df_balanced['f2_td_avg']
    df_balanced['td_acc_diff'] = df_balanced['f1_td_acc_f'] - df_balanced['f2_td_acc_f']
    df_balanced['td_def_diff'] = df_balanced['f1_td_def_f'] - df_balanced['f2_td_def_f']
    df_balanced['sub_avg_diff'] = df_balanced['f1_sub_avg'] - df_balanced['f2_sub_avg']
    df_balanced['experience_diff'] = df_balanced['f1_experience'] - df_balanced['f2_experience']
    df_balanced['win_streak_diff'] = df_balanced['f1_win_streak'] - df_balanced['f2_win_streak']
    df_balanced['recent_form_diff'] = df_balanced['f1_recent_form'] - df_balanced['f2_recent_form']
    df_balanced['days_since_last_diff'] = df_balanced['f1_days_since_last'] - df_balanced['f2_days_since_last']
    df_balanced['age_diff'] = df_balanced['f1_age'] - df_balanced['f2_age']
    df_balanced['is_title_fight'] = df_balanced['is_title_fight'].astype(int)

    # Stance matchup
    def stance_matchup(s1, s2):
        if pd.isna(s1) or pd.isna(s2):
            return 'unknown'
        s1, s2 = s1.strip().lower(), s2.strip().lower()
        if s1 == s2: return 'same'
        if set([s1, s2]) == {'orthodox', 'southpaw'}: return 'ortho_vs_south'
        return 'other'

    df_balanced['stance_matchup'] = df_balanced.apply(
        lambda r: stance_matchup(r['f1_stance'], r['f2_stance']), axis=1)
    stance_dummies = pd.get_dummies(df_balanced['stance_matchup'], prefix='stance')
    weight_dummies = pd.get_dummies(df_balanced['weight_class'], prefix='wc')
    df_balanced = pd.concat([df_balanced, stance_dummies, weight_dummies], axis=1)

    return df_balanced

def train():
    engine = get_engine()
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("ufc_fight_prediction")

    print("Building dataset...")
    df = build_dataset(engine)

    feature_cols = [
        'height_diff', 'reach_diff', 'slpm_diff', 'sapm_diff',
        'str_acc_diff', 'str_def_diff', 'td_avg_diff', 'td_acc_diff',
        'td_def_diff', 'sub_avg_diff', 'experience_diff', 'win_streak_diff',
        'recent_form_diff', 'days_since_last_diff', 'age_diff', 'is_title_fight',
        'stance_same', 'stance_ortho_vs_south', 'stance_other', 'stance_unknown',
        'wc_Bantamweight', 'wc_Catch Weight', 'wc_Featherweight', 'wc_Flyweight',
        'wc_Heavyweight', 'wc_Light Heavyweight', 'wc_Lightweight', 'wc_Middleweight',
        'wc_Open Weight', 'wc_Super Heavyweight', 'wc_Welterweight',
        "wc_Women's Bantamweight", "wc_Women's Featherweight",
        "wc_Women's Flyweight", "wc_Women's Strawweight"
    ]

    # Only keep feature cols that exist
    feature_cols = [f for f in feature_cols if f in df.columns]
    target_col = 'fighter_1_won'

    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X = pd.DataFrame(imputer.fit_transform(df[feature_cols]), columns=feature_cols)
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    with mlflow.start_run(run_name="xgboost_weekly_retrain") as run:
        params = {
            "model_type": "XGBoost",
            "n_estimators": 700,
            "max_depth": 4,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42
        }
        mlflow.log_params(params)

        xgb = XGBClassifier(**{k: v for k, v in params.items() if k != "model_type"})
        xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        train_acc = accuracy_score(y_train, xgb.predict(X_train))
        test_acc = accuracy_score(y_test, xgb.predict(X_test))
        test_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
        test_loss = log_loss(y_test, xgb.predict_proba(X_test))

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_log_loss", test_loss)

        mlflow.xgboost.log_model(xgb, "model")

        # Register new version
        model_uri = f"runs:/{run.info.run_id}/model"
        registered = mlflow.register_model(model_uri, "ufc_fight_predictor")
        new_version = registered.version

        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        print(f"Test AUC:       {test_auc:.4f}")
        print(f"Test Log Loss:  {test_loss:.4f}")
        print(f"new_version:{new_version}")

if __name__ == "__main__":
    train()
