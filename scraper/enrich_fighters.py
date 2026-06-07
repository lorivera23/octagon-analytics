import asyncio
from contextlib import closing
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from db import get_db

EXPECTED_STATS = {"SLPM", "STR. ACC.", "SAPM", "STR. DEF", "TD AVG.",
                  "TD ACC.", "TD DEF.", "SUB. AVG."}

def parse_fighter_stats(html):
    soup = BeautifulSoup(html, "html.parser")
    items = soup.select("li.b-list__box-list-item")

    if not items:
        raise ValueError(
            "parse_fighter_stats found 0 list items — page layout or selector changed"
        )

    stats = {}
    for item in items:
        parts = [p.strip() for p in item.get_text(separator="|").split("|") if p.strip()]
        if len(parts) >= 2:
            label = parts[0].rstrip(":").upper()
            value = parts[1].strip()
            if value and value != "--":
                stats[label] = value

    missing = EXPECTED_STATS - stats.keys()
    if missing:
        raise ValueError(
            f"parse_fighter_stats parsed page but missing expected stats: {sorted(missing)}"
        )


    return stats

def update_fighter(conn, fighter_id, stats):
    # fighter is one atomic unit, leave the commit responsibility to this function
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
    with closing(get_db()) as conn:
        fighters = get_unenriched_fighters(conn)
        print(f"Enriching {len(fighters)} fighters...")

       
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            failed = []
            for i, (fighter_id, name, url) in enumerate(fighters):
                try:
                    await page.goto(url, timeout=60000)
                    await page.wait_for_selector("li.b-list__box-list-item", timeout=15000)
                    html = await page.content()

                    stats = parse_fighter_stats(html)
                    update_fighter(conn, fighter_id, stats)

                    print(f"[{i+1}/{len(fighters)}] {name} → {stats}")

                except Exception as e:
                    conn.rollback()        # clears aborted-transaction state so the loop can continue
                    failed.append(name)
                    print(f"[{i+1}/{len(fighters)}] {name} → Error: {e}")
                
                await asyncio.sleep(1)
          
            await browser.close()

        if fighters and len(failed) == len(fighters):
            raise RuntimeError(
                f"All {len(fighters)} fighters failed to enrich — systemic failure. "
                f"Examples: {failed[:3]}"
            )
        if failed:
            print(f"WARNING: {len(failed)}/{len(fighters)} fighters failed: {failed}")
  
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
