import asyncio
import re
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import psycopg2
import time
import os

# ufcstats is gated by a JS proof-of-work challenge (nonce + SHA-256 with 2-zero-hex-prefix, POST to /__c).
# Playwright is justified — requests+headers cannot pass it.
# Optional optimization: solve the PoW in Python and use a requests.Session to drop the browser dependency (faster, but brittle to
# challenge changes).


DB_CONFIG = {
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"], # hard-cdoed, put this in env files before commiting anywhere
    "host": os.environ["DB_HOST"]
}

BASE_URL = "http://ufcstats.com"

def get_db():
    return psycopg2.connect(**DB_CONFIG)

def extract_id_from_url(url):
    return url.rstrip("/").split("/")[-1]

async def scrape_with_browser(url, selector, page):
    # fresh page per fetch + retry both belong in scrape_with_browser — change its signature from page to browser, create/close the page
    # inside with a retry loop, and thread browser through get_events/get_fights/orchestrators. 
    # Isolates wedged tabs AND handles transient nav failures in one place
    # read run_incremental for more information on the problem 

    # test the refactor with fault injection, not just happy path — bad URL + tiny timeout to force the retry/raise path; 
    # assert loop continues past a failed event and pages always close (watch the finally).
    # Use a scratch DB since ON CONFLICT DO NOTHING hides re-run effects. (more on fault injection in claude chat)
    await page.goto(url, timeout=60000)
    await page.wait_for_selector(selector, timeout=30000)
    return await page.content()

# hard coded css columns to be aware of for future fixes
# if tr.b-statistics__table-row gets changed at any point scrape with browswer will timeout 
# this is a loud failure, and the task will go red in airflow, not fill with garbage.


async def get_events(page):
    print("Scraping events...")
    html = await scrape_with_browser(
        f"{BASE_URL}/statistics/events/completed?page=all",
        "tr.b-statistics__table-row",
        page
    )
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tr.b-statistics__table-row")

    events = []
    for row in rows:

        # this could fail silently, beautiful soup does not raise errors when a css selector is not found in either the select case
        # or the select_one case, need to add a raise error (not assert, as we do not want a silent fail) block before to return to 
        # check if events gets populated
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

    # raise error hear, if events is null. consider not just a non-zero check, see if len is less that 700 or something

    return events

async def get_fights(event_url, page):
    # FRAGILITY — positional column parsing:
# 1. Guard is `len(cols) < 8` but code reads cols[8] and cols[9] (needs 10 cols).
#    Off-by guard: an 8- or 9-col row passes the check then IndexErrors.
#    IndexError is caught by run_incremental's per-event try -> whole event silently skipped.
#    Fix: align guard to actual max index used (< 10), OR parse by header name not position.
# 2. WORSE CASE — if ufcstats REORDERS/INSERTS a column (same count, shifted meaning),
#    every read still succeeds and wrong-but-plausible data is written silently
#    (method->round, etc). No length guard can catch a reorder; count is unchanged.
#    Only real defense is VALUE validation: round parses as int 1-5, method in known set,
#    time matches M:SS. Content validation, not structure validation.
# Interview point: distinguish validating SHAPE (col count) from validating MEANING (cell values).
    html = await scrape_with_browser(
        event_url,
        "tr.b-fight-details__table-row",
        page
    )
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tr.b-fight-details__table-row")

    fights = []
    for row in rows:
	# same techincal problem as get_fights(), but other things to consider on one event being broken
        # should one bad event kill the entire run, or get logged and skipped while the run continues?
        cols = row.select("td.b-fight-details__table-col")
        if len(cols) < 8:
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

        # Green flag = fighter 1 won, check text for "win"
        flag_text = cols[0].select_one("i.b-flag__text")
        flag_value = flag_text.text.strip() if flag_text else None

        if flag_value == "win":
            winner_id = fighter_1_id
        else:
            # Check if there's a second flag for fighter 2
            all_flags = cols[0].select("i.b-flag__text")
            if len(all_flags) > 1 and all_flags[1].text.strip() == "win":
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

    # either assert error here or log and skip 

    return fights
    
def save_event(conn, event):
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
    conn.commit()

def save_fighter(conn, fighter):
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
    conn.commit()

def save_fight(conn, fight, event_id):

    # potential problem
    # if for some reason, say a database contraint throws an exception when trying to insert when going through an event
    # some fights could get saved for a given event, then hit a constraint problem on the database, leaving the rest of the fights out
    # This function gets called weekly in a try / except statement in run_incremental, and will fail silentely never letting me know
    # we would miss out on other fighs with out knowing it
    # solution: remove conn.commit from inside all the save_ functions, commit once in run_incremental, after each event's fights
    #           suceed, conn.rollback() in the except.
    # this makes the 'event exists' logic in get_existing_events() equivilant to 'card complete', which is what is already is assuming
    
    # note: This change is somewhat of a contract change and removing commits from save_ functions now assumes an open transaction 
    #       and that another function will commit its changes, leave a comment on functions when documenting so I do not call save_
    #       and then be suprised when nothing persists
    
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
    conn.commit()
    

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
    conn = get_db()
    existing = get_existing_events(conn)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # using one page for the entire scrape could be problementaic, if we run into one problem/error we could be left with an
        # unhealthy page state without knowing it, and it would silently fail.
        # fix is going to be a refractor of who orchestrates pages (getting moved to scrape_with_broswer, check there for further 
        # changes)
        page = await browser.new_page()

        all_events = await get_events(page)
        new_events = [e for e in all_events if e["event_id"] not in existing]

        print(f"Found {len(new_events)} new events")

        for i, event in enumerate(new_events):
            print(f"[{i+1}/{len(new_events)}] Scraping: {event['name']}")
            # no op function, upcoming.py will scrape events and save them
            save_event(conn, event)

            try:
                fights = await get_fights(event["url"], page)
                for fight in fights:
                    if not fight["fight_id"]:
                        continue
                    save_fighter(conn, fight["fighter_1"])
                    save_fighter(conn, fight["fighter_2"])
                    save_fight(conn, fight, event["event_id"])
                print(f"  → {len(fights)} fights saved")
            except Exception as e:
                print(f"  → Error: {e}")

            await asyncio.sleep(2)

        await browser.close()
    # conn.close() at the bottom is skipped if anything above raises (e.g. get_events timeout) — leaks the connection. 
    # Fix with contextlib.closing(get_db()) for guaranteed close. Do NOT use bare with conn: expecting it to close — psycopg2's with 
    # manages the TRANSACTION (commit/rollback), not the connection. Keep close (closing) and commit (per-event) as separate concerns.

    conn.close()
    return len(new_events)

async def main():
    conn = get_db()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # same problem as run_incremental's page, fix this one too while we are at it
        page = await browser.new_page()

        events = await get_events(page)
        print(f"Found {len(events)} events")

        for i, event in enumerate(events):
            print(f"[{i+1}/{len(events)}] Scraping: {event['name']}")

            save_event(conn, event)

            try:
                fights = await get_fights(event["url"], page)
                for fight in fights:
                    if not fight["fight_id"]:
                        continue
                    save_fighter(conn, fight["fighter_1"])
                    save_fighter(conn, fight["fighter_2"])
                    save_fight(conn, fight, event["event_id"])

                print(f"  → {len(fights)} fights saved")
            except Exception as e:
                print(f"  → Error: {e}")

            # Be polite to the server
            await asyncio.sleep(2)

        await browser.close()

    conn.close()
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
