import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import psycopg2
import time
import os

# change these to environment variables
# also this + get_db logic is duplicated from scraper.py, make a shared db.py / config module for both config and get db
DB_CONFIG = {
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "host": os.environ["DB_HOST"]
}


def get_db():
    return psycopg2.connect(**DB_CONFIG)

def parse_fighter_stats(html):
    soup = BeautifulSoup(html, "html.parser")
    stats = {}

    # this line could quietly fail, which would give an empty list that is never caught, potentially never enriching fighters if this
    # css selector were to change and even worse, if a stat other than splm is missing it would consider it enriched and we would never
    # know, add a check to see if this items list ever gets populated
    items = soup.select("li.b-list__box-list-item")
    for item in items:
        parts = [p.strip() for p in item.get_text(separator="|").split("|") if p.strip()]
        if len(parts) >= 2:
            label = parts[0].rstrip(":").upper()
            value = parts[1].strip()
            if value and value != "--":
                stats[label] = value

    # Need to add a check here ensuring that our string keys from update_fighter() match whatever we grab
    # make a list of expected, and raise if there are any missing

    return stats

def update_fighter(conn, fighter_id, stats):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE fighters SET
                height  = %s,
                reach   = %s,
                stance  = %s,
                dob     = %s,
                weight  = %s,
                slpm    = %s,
                str_acc = %s,
                sapm    = %s,
                str_def = %s,
                td_avg  = %s,
                td_acc  = %s,
                td_def  = %s,
                sub_avg = %s
            WHERE fighter_id = %s
        """, (
            stats.get("HEIGHT"),
            stats.get("REACH"),
            stats.get("STANCE"),
            stats.get("DOB"),
            stats.get("WEIGHT"),
            stats.get("SLPM"),
            stats.get("STR. ACC."),
            stats.get("SAPM"),
            stats.get("STR. DEF"),
            stats.get("TD AVG."),
            stats.get("TD ACC."),
            stats.get("TD DEF."),
            stats.get("SUB. AVG."),
            fighter_id
        ))
    conn.commit()

def get_unenriched_fighters(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT fighter_id, name, url FROM fighters
            WHERE slpm IS NULL AND url IS NOT NULL
        """)
        return cur.fetchall()

async def main():
    conn = get_db()
    fighters = get_unenriched_fighters(conn)
    print(f"Enriching {len(fighters)} fighters...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for i, (fighter_id, name, url) in enumerate(fighters):
            # this try statement would catch any errors raised by parse_fighter_stats (raise checks to be added), making them useless
            # Add a counter that raises after the loop failure rate is high
            try:
                await page.goto(url, timeout=60000)
                await page.wait_for_selector("li.b-list__box-list-item", timeout=15000)
                html = await page.content()

                stats = parse_fighter_stats(html)
                update_fighter(conn, fighter_id, stats)

                print(f"[{i+1}/{len(fighters)}] {name} → {stats}")

            except Exception as e:
                print(f"[{i+1}/{len(fighters)}] {name} → Error: {e}")

            await asyncio.sleep(1)
        # similar lifecycle issue from scraper.py, commits live inside the update function, and we do one open and close, which
        # could skip erros
        await browser.close()

    conn.close()
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
