import asyncio
from playwright.async_api import async_playwright
from contextlib import closing
from bs4 import BeautifulSoup
from db import get_db

# ufcstats is gated by a JS proof-of-work challenge (nonce + SHA-256 with 2-zero-hex-prefix, POST to /__c).
# Playwright is justified — requests+headers cannot pass it.
# Optional optimization: solve the PoW in Python and use a requests.Session to drop the browser dependency (faster, but brittle to
# challenge changes).


BASE_URL = "http://ufcstats.com"


def extract_id_from_url(url):
    return url.rstrip("/").split("/")[-1]

async def scrape_with_browser(url, selector, browser, retries=2):
    """Fetch a fully-rendered page. Creates a fresh page per call (isolates wedged tabs),
    retries transient navigation failures, and always closes the page."""
    last_err = None
    for attempt in range(retries + 1):
        page = await browser.new_page()
        try:
            await page.goto(url, timeout=60000)
            await page.wait_for_selector(selector, timeout=30000)
            return await page.content()
        except Exception as e:
            last_err = e
            print(f"  fetch attempt {attempt+1}/{retries+1} failed for {url}: {e}")
        finally:
            await page.close()
    raise last_err

async def get_events(browser):
    print("Scraping events...")
    html = await scrape_with_browser(
        f"{BASE_URL}/statistics/events/completed?page=all",
        "a.b-link.b-link_style_black",
        browser
    )
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tr.b-statistics__table-row")

    events = []
    for row in rows:

        link = row.select_one("a.b-link.b-link_style_black")
        date_span = row.select_one("span.b-statistics__date")
        location_td = row.select_one("td.b-statistics__table-col.b-statistics__table-col_style_big-top-padding")

        if link:
            events.append({
                "event_id": extract_id_from_url(link["href"]),
                "name": link.text.strip(),
                "url": link["href"],
                "date": date_span.text.strip() if date_span else None,
                "location": location_td.text.strip() if location_td else None
            })

    if not events:
        raise ValueError(
            f"get_events parsed 0 events from {len(rows)} table rows — "
            f"page layout or anti-bot challenge likely changed"
        )
   
    return events

# Some historical decision fights were previously scraped with winner_id=None.
# Legitimate no-winner outcomes are handled downstream; decision rows with a
# null winner are treated as data-quality issues and excluded from training.
async def get_fights(event_url, browser):
    html = await scrape_with_browser(
        event_url,
        "tr.b-fight-details__table-row",
        browser
    )
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tr.b-fight-details__table-row")

    fights = []
    for row in rows:
        cols = row.select("td.b-fight-details__table-col")
        if len(cols) < 10:
            continue

        fighter_links = cols[1].select("a.b-link")
        if len(fighter_links) < 2:
            continue

        fighter_1_id = extract_id_from_url(fighter_links[0]["href"])
        fighter_2_id = extract_id_from_url(fighter_links[1]["href"])
        fighter_1_name = fighter_links[0].text.strip()
        fighter_2_name = fighter_links[1].text.strip()

        # Fight URL and winner flag are in col 0
        flag_link = cols[0].select_one("a.b-flag")
        fight_url = flag_link["href"] if flag_link else None
        fight_id = extract_id_from_url(fight_url) if fight_url else None

        # Parse the result flags for both fighters and map the winner explicitly.
        result_flags = [
            flag.text.strip().lower()
            for flag in cols[0].select("i.b-flag__text")
        ]

        fighter_1_result = result_flags[0] if len(result_flags) > 0 else None
        fighter_2_result = result_flags[1] if len(result_flags) > 1 else None

        if fighter_1_result == "win":
            winner_id = fighter_1_id
        elif fighter_2_result == "win":
            winner_id = fighter_2_id
        else:
            winner_id = None

        fights.append({
            "fight_id": fight_id,
            "fight_url": fight_url,
            "fighter_1": {
                "name": fighter_1_name,
                "url": fighter_links[0]["href"],
                "fighter_id": fighter_1_id
            },
            "fighter_2": {
                "name": fighter_2_name,
                "url": fighter_links[1]["href"],
                "fighter_id": fighter_2_id
            },
            "winner_id": winner_id,
            "method": cols[7].text.strip() if len(cols) > 7 else None,
            "round": cols[8].text.strip() if len(cols) > 8 else None,
            "time": cols[9].text.strip() if len(cols) > 9 else None,
            "weight_class": cols[6].text.strip() if len(cols) > 6 else None,
        })

    # check if rows were on the page but no fights were parts, layout could change 
    if rows and not fights:
        raise ValueError(
            f"get_fights parsed 0 fights from {len(rows)} rows on {event_url} — "
            f"fight table layout likely changed"
        )
    
    return fights

  
    
def save_event(conn, event):
    # Note: no conn.commit() here, caller (run_incremental) owns the transaction and commits per-event 
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO events (event_id, name, date, location, url)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (event_id) DO NOTHING
        """, (
            event["event_id"],
            event["name"],
            event["date"],
            event["location"],
            event["url"]
        ))
  

def save_fighter(conn, fighter):
    # Note: no conn.commit() here, caller (run_incremental) owns the transaction and commits per-event 
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO fighters (fighter_id, name, url)
            VALUES (%s, %s, %s)
            ON CONFLICT (fighter_id) DO NOTHING
        """, (
            fighter["fighter_id"],
            fighter["name"],
            fighter["url"]
        ))
    

def save_fight(conn, fight, event_id):  
    # Note: no conn.commit() here, caller (run_incremental) owns the transaction and commits per-event 
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO fights (fight_id, event_id, fighter_1_id, fighter_2_id, winner_id, method, round, time, weight_class)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (fight_id) DO NOTHING
        """, (
            fight["fight_id"],
            event_id,
            fight["fighter_1"]["fighter_id"],
            fight["fighter_2"]["fighter_id"],
            fight["winner_id"],
            fight["method"],
            fight["round"],
            fight["time"],
            fight["weight_class"]
        ))
    
    

def get_existing_events(conn):
    with conn.cursor() as cur:
        # Only consider events that already have fights scraped
        cur.execute("""
            SELECT DISTINCT e.event_id 
            FROM events e
            JOIN fights f ON e.event_id = f.event_id
        """)
        return {row[0] for row in cur.fetchall()}



async def run_incremental():
    """Only scrape events not already in the database - used by Airflow"""
    with closing(get_db()) as conn:

        existing = get_existing_events(conn)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
           
            all_events = await get_events(browser)

            if len(all_events) < len(existing):
                raise ValueError(
                    f"Scraped {len(all_events)} events but DB already has {len(existing)} "
                    f"completed events — scrape likely incomplete (partial page load?)"
                )

            new_events = [e for e in all_events if e["event_id"] not in existing]

            print(f"Found {len(new_events)} new events")

            failed_events = []

            for i, event in enumerate(new_events):
                print(f"[{i+1}/{len(new_events)}] Scraping: {event['name']}")
                
                try:
                    save_event(conn, event)
                    fights = await get_fights(event["url"], browser)
                    for fight in fights:
                        if not fight["fight_id"]:
                            continue
                        save_fighter(conn, fight["fighter_1"])
                        save_fighter(conn, fight["fighter_2"])
                        save_fight(conn, fight, event["event_id"])
                    conn.commit() # commit per-event, so a failure on one event doesn't roll back the whole run
                    print(f"  → {len(fights)} fights saved")
                except Exception as e:
                    conn.rollback() # rollback any partial changes from this event, move on to the next one
                    failed_events.append(event["name"])
                    print(f"  → Error on {event['name']}: {e}")

                await asyncio.sleep(2)

            await browser.close()
        
        # raise error if all events failed, otherwise log a warning with the count and examples of failed events.
        if new_events and len(failed_events) == len(new_events):
            raise RuntimeError(
                f"All {len(new_events)} events failed to scrape — systemic failure. "
                f"Examples: {failed_events[:3]}"
            )
        if failed_events:
            print(f"WARNING: {len(failed_events)}/{len(new_events)} events failed: {failed_events}")

    return len(new_events)

async def main():
    with closing(get_db()) as conn:

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
        
            events = await get_events(browser)
            print(f"Found {len(events)} events")

            failed_events = []

            for i, event in enumerate(events):
                print(f"[{i+1}/{len(events)}] Scraping: {event['name']}")

                try:
                    save_event(conn, event)
                    fights = await get_fights(event["url"], browser)
                    for fight in fights:
                        if not fight["fight_id"]:
                            continue
                        save_fighter(conn, fight["fighter_1"])
                        save_fighter(conn, fight["fighter_2"])
                        save_fight(conn, fight, event["event_id"])
                    conn.commit() # commit per-event, so a failure on one event doesn't roll back the whole run
                    print(f"  → {len(fights)} fights saved")
                except Exception as e:
                    conn.rollback() # rollback any partial changes from this event, move on to the next one
                    failed_events.append(event["name"])
                    print(f"  → Error on {event['name']}: {e}")

                # Be polite to the server
                await asyncio.sleep(2)

            await browser.close()

        # raise error if all events failed, otherwise log a warning with the count and examples of failed events.
        if events and len(failed_events) == len(events):
            raise RuntimeError(
                f"All {len(events)} events failed to scrape — systemic failure. "
                f"Examples: {failed_events[:3]}"
            )
        if failed_events:
            print(f"WARNING: {len(failed_events)}/{len(events)} events failed: {failed_events}")

    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
