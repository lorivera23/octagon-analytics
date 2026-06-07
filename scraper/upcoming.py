import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from db import get_db
import json
import os



BASE_URL = "http://ufcstats.com"
API_URL = os.environ.get("API_URL", "http://localhost:8000")



def extract_id(url):
    return url.rstrip("/").split("/")[-1]

def get_existing_predictions(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT fight_id FROM predictions WHERE fight_id IS NOT NULL")
        return {row[0] for row in cur.fetchall()}

# I am going to take this function out entirely as it is a standalone step in the DAG, and does not really belong in this script
# this require taking the call out the main loop here, and only having the DAG use it (this may end up requiring a rework of the dag 
# script currently in place depending on what it calls, because right now I think both the dag script and the main function call
# score_pending
# Also worth noting that once this gets taken out of this script, it will need it's own databse configurations and if name = main call
# Take into consideration the database connection lifecycle problem we have been running into with the other scripts 
# add an invariant check to make sure that is up_coming = false + null winner means draw or no contest
# 

def score_pending_predictions(conn):
    """Score predictions where fight results are now available"""
    with conn.cursor() as cur:
        # Find predictions that are still upcoming but fight is now in fights table
        cur.execute("""
            SELECT p.prediction_id, p.fight_id, p.predicted_winner_id,
                   f.winner_id, p.fighter_1_name, p.fighter_2_name
            FROM predictions p
            JOIN fights f ON p.fight_id = f.fight_id
            WHERE p.is_upcoming = TRUE
            AND (f.winner_id IS NOT NULL
                OR f.method IN ('CNC', 'NC, 'DQ'))
        """)
        rows = cur.fetchall()

    scored = 0
    for pred_id, fight_id, predicted_winner, actual_winner, f1_name, f2_name in rows:
        if actual_winner is None:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE predictions SET
                        is_upcoming = FALSE,
                        correct = NULL
                    WHERE prediction_id = %s
                """, (pred_id,))
            conn.commit()
            print(f"⬜ {f1_name} vs {f2_name} — No Contest, removed from upcoming")
            scored += 1
            continue


        correct = predicted_winner == actual_winner
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE predictions SET
                    actual_winner_id = %s,
                    correct = %s,
                    is_upcoming = FALSE
                WHERE prediction_id = %s
            """, (actual_winner, correct, pred_id))
        conn.commit()
        result = "✓" if correct else "✗"
        print(f"{result} {f1_name} vs {f2_name} — predicted {'correctly' if correct else 'incorrectly'}")
        scored += 1

    print(f"Scored {scored} predictions")
    return scored

async def get_upcoming_events(page):
    print("Scraping upcoming events...")
    await page.goto(f"{BASE_URL}/statistics/events/upcoming", timeout=60000)
    await page.wait_for_selector("tr.b-statistics__table-row", timeout=30000)
    html = await page.content()

    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tr.b-statistics__table-row")

    events = []
    for row in rows:
        link = row.select_one("a.b-link.b-link_style_black")
        date_span = row.select_one("span.b-statistics__date")
        location_td = row.select_one("td.b-statistics__table-col.b-statistics__table-col_style_big-top-padding")

        if link:
            events.append({
                "event_id": extract_id(link["href"]),
                "name": link.text.strip(),
                "url": link["href"],
                "date": date_span.text.strip() if date_span else None,
                "location": location_td.text.strip() if location_td else None
            })

    # events should get populated, but rows=soup.select() could fail silently without raising, raise an error here if events is empty
    return events


async def get_upcoming_fights(event_url, page):
    await page.goto(event_url, timeout=60000)
    await page.wait_for_selector("tr.b-fight-details__table-row", timeout=30000)
    html = await page.content()

    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tr.b-fight-details__table-row")

    fights = []
    for row in rows:
        cols = row.select("td.b-fight-details__table-col")
        # same column guard mismatch we saw in scraper.py, fix this as well
        if len(cols) < 6:
            continue

        fighter_links = cols[1].select("a.b-link")
        if len(fighter_links) < 2:
            continue

        fight_url = row.get("data-link")
        fight_id = extract_id(fight_url) if fight_url else None

        fights.append({
            "fight_id": fight_id,
            "fighter_1_name": fighter_links[0].text.strip(),
            "fighter_1_url": fighter_links[0]["href"],
            "fighter_2_name": fighter_links[1].text.strip(),
            "fighter_2_url": fighter_links[1]["href"],
            "weight_class": cols[6].text.strip() if len(cols) > 6 else None,
        })
    # check if fights gets populated as soup does not raise errors 
    return fights

def save_upcoming_event(conn, event):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO events (event_id, name, date, location, url)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING
            """, (event["event_id"], event["name"], event["date"],
                  event["location"], event["url"]))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error saving event: {e}")

def predict_and_store(conn, fight, event_id, existing_predictions):
    if not fight["fight_id"] or fight["fight_id"] in existing_predictions:
        return False

    try:
        response = requests.post(f"{API_URL}/predict", json={
            "fighter_1": fight["fighter_1_name"],
            "fighter_2": fight["fighter_2_name"],
            "weight_class": fight["weight_class"],
            "event_id": event_id
        }, timeout=10)

        # add a check for a 500 connection error, this would mean that the api service is down, and would need a error raised
        # not skipped
        # also add a check to make sure that the fight_id, fighter_1_name, and fighter_name are all actually caught when stitched
        if response.status_code == 200:
            result = response.json()
            print(f"  {fight['fighter_1_name']} vs {fight['fighter_2_name']} → {result['predicted_winner']} ({result['fighter_1_win_probability']:.0%} / {result['fighter_2_win_probability']:.0%})")

            # 2 existing orphan rows need manual cleanup, fight id fix prevents new ones
            # this is a very fragile step and would work a lot cleaner if we just pass the fight id to the /predict endpoint 
            # Update fight_id on the prediction just stored
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE predictions SET fight_id = %s
                    WHERE prediction_id = (
                        SELECT prediction_id FROM predictions
                        WHERE fighter_1_name = %s 
                        AND fighter_2_name = %s
                        AND fight_id IS NULL
                        ORDER BY predicted_at DESC
                        LIMIT 1
                    )
                """, ((fight["fight_id"], fight["fighter_1_name"], fight["fighter_2_name"]))) # double parentheses are not needed but do not break anything
            conn.commit()
            return True
        else:
            print(f"  Skipped {fight['fighter_1_name']} vs {fight['fighter_2_name']} — fighter not in DB")
            return False

    except Exception as e:
        conn.rollback()
        print(f"  Error predicting {fight['fighter_1_name']} vs {fight['fighter_2_name']}: {e}")
        return False

async def main():
    conn = get_db()

    # Step 1: Score any pending predictions
    print("=== Scoring pending predictions ===")
    score_pending_predictions(conn)

    # Step 2: Scrape upcoming events and make predictions
    print("\n=== Scraping upcoming events ===")
    existing_predictions = get_existing_predictions(conn)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        events = await get_upcoming_events(page)
        print(f"Found {len(events)} upcoming events")

        for event in events:
            print(f"\n{event['name']} ({event['date']})")
            save_upcoming_event(conn, event)

            try:
                fights = await get_upcoming_fights(event["url"], page)
                predicted = 0
                for fight in fights:
                    if predict_and_store(conn, fight, event["event_id"], existing_predictions):
                        predicted += 1
                print(f"  → {predicted} predictions stored")
            except Exception as e:
                print(f"  → Error: {e}")

            await asyncio.sleep(2)

        await browser.close()

    conn.close()
    print("\nDone.")

if __name__ == "__main__":
    asyncio.run(main())
