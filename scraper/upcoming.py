import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from contextlib import closing
from db import get_db
import requests
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


async def get_upcoming_events(page):
    print("Scraping upcoming events...")
    await page.goto(f"{BASE_URL}/statistics/events/upcoming", timeout=60000)
    await page.wait_for_selector("a.b-link.b-link_style_black", timeout=30000)
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

    if not events:
        raise ValueError(
            f"get_upcoming_events parsed 0 events from {len(rows)} rows — layout/PoW changed"
        )

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
        if len(cols) < 7:
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

    if rows and not fights:
        raise ValueError(
            f"get_upcoming_fights parsed 0 fights from {len(rows)} rows on {event_url} — layout changed"
        ) 
          
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
        return "skipped"

    try:
        response = requests.post(f"{API_URL}/predict", json={
            "fighter_1": fight["fighter_1_name"],
            "fighter_2": fight["fighter_2_name"],
            "weight_class": fight["weight_class"],
            "event_id": event_id,
            "fight_id": fight["fight_id"],
        }, timeout=10)

        # add a check for a 500 connection error, this would mean that the api service is down, and would need a error raised
        # not skipped
        # also add a check to make sure that the fight_id, fighter_1_name, and fighter_name are all actually caught when stitched
        if response.status_code == 200:
            result = response.json()
            print(f"  {fight['fighter_1_name']} vs {fight['fighter_2_name']} → {result['predicted_winner']} ({result['fighter_1_win_probability']:.0%} / {result['fighter_2_win_probability']:.0%})")
            return "predicted"
        elif response.status_code == 404:
            print(f"  Skipped {fight['fighter_1_name']} vs {fight['fighter_2_name']} — fighter not in DB")
            return "skipped"
        else:
            # 500 / unexpected — API problem, not a missing fighter. Surface it.
            raise RuntimeError(
                f"API returned {response.status_code} for "
                f"{fight['fighter_1_name']} vs {fight['fighter_2_name']}"
            )

    except Exception as e:
        conn.rollback()
        print(f"  Error predicting {fight['fighter_1_name']} vs {fight['fighter_2_name']}: {e}")
        return "error"

async def run_upcoming():
    with closing(get_db()) as conn:

        print("\n=== Scraping upcoming events ===")
        existing_predictions = get_existing_predictions(conn)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            events = await get_upcoming_events(page)
            print(f"Found {len(events)} upcoming events")

            failed_events = []
            predict_attempts = 0
            predict_errors = 0

            for event in events:
                print(f"\n{event['name']} ({event['date']})")
                save_upcoming_event(conn, event)

                try:
                    fights = await get_upcoming_fights(event["url"], page)
                    predicted = 0
                    for fight in fights:
                        result = predict_and_store(conn, fight, event["event_id"], existing_predictions)
                        if result == "predicted":
                            predicted += 1
                        elif result == "error":
                            predict_errors += 1
                        if result in ("predicted", "error"):
                            predict_attempts += 1
                        
                    print(f"  → {predicted} predictions stored")
                except Exception as e:
                    failed_events.append(event["name"])
                    print(f"  → Error on {event['name']}: {e}")

                await asyncio.sleep(2)

            await browser.close()

        if events and len(failed_events) == len(events):
            raise RuntimeError(
                f"All {len(events)} upcoming events failed to scrape — systemic failure. "
                f"Examples: {failed_events[:3]}"
            )
        if predict_attempts and predict_errors == predict_attempts:
            raise RuntimeError(
                f"All {predict_attempts} prediction attempts errored — API likely down."
            )
        if failed_events:
            print(f"WARNING: {len(failed_events)}/{len(events)} events failed: {failed_events}")

        print("\nDone.")

if __name__ == "__main__":
    asyncio.run(run_upcoming())