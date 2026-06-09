from contextlib import closing
from db import get_db

# Null-winner methods that are LEGITIMATELY without a winner (fight happened, no win recorded).
# Confirmed against `SELECT DISTINCT split_part(method, E'\n', 1) FROM fights WHERE winner_id IS NULL`.
# Decisions (*-DEC) with a null winner are a DATA BUG, not a no-contest — handled separately below.
NO_WINNER_METHODS = {"CNC", "Overturned", "Other"}


def score_pending_predictions(conn):
    """Score predictions whose fights now have results.
    A prediction is scoreable once its fight exists in `fights` (presence = the fight happened).
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT p.prediction_id, p.fight_id, p.predicted_winner_id,
                   f.winner_id, f.method, p.fighter_1_name, p.fighter_2_name
            FROM predictions p
            JOIN fights f ON p.fight_id = f.fight_id
            WHERE p.is_upcoming = TRUE
        """)
        rows = cur.fetchall()

    scored = 0
    data_bugs = 0
    for pred_id, fight_id, predicted_winner, actual_winner, method, f1, f2 in rows:
        method_clean = (method or "").split("\n")[0].strip()

        if actual_winner is not None:
            # Someone won (decision, finish, or DQ — DQ has a winner_id, so it scores normally).
            correct = (predicted_winner == actual_winner)
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE predictions SET
                        actual_winner_id = %s, correct = %s, is_upcoming = FALSE
                    WHERE prediction_id = %s
                """, (actual_winner, correct, pred_id))
            conn.commit()
            print(f"{'✓' if correct else '✗'} {f1} vs {f2} — predicted {'correctly' if correct else 'incorrectly'}")
            scored += 1

        elif method_clean in NO_WINNER_METHODS:
            # Legitimate no-contest / overturned — resolved but not scoreable.
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE predictions SET correct = NULL, is_upcoming = FALSE
                    WHERE prediction_id = %s
                """, (pred_id,))
            conn.commit()
            print(f"⬜ {f1} vs {f2} — {method_clean}, no winner, removed from upcoming")
            scored += 1

        else:
            # Null winner but method implies one (e.g. M-DEC/U-DEC) — DATA BUG, not a no-contest.
            # Leave is_upcoming = TRUE so it isn't silently mislabeled; surface for re-scrape.
            conn.rollback()
            data_bugs += 1
            print(f"WARNING: {f1} vs {f2} ({method_clean}) has null winner but method implies one — "
                  f"left unscored, fight_id={fight_id}")

    print(f"Scored {scored} predictions ({data_bugs} skipped as suspected data bugs)")
    return scored


if __name__ == "__main__":
    with closing(get_db()) as conn:
        score_pending_predictions(conn)