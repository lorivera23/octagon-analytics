from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
import pendulum
import os

MLFLOW_URI = os.environ["MLFLOW_URI"]

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ufc_weekly_pipeline",
    schedule_interval="0 6 * * MON",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args=default_args,
    tags=["ufc", "mlops"],
    description="Weekly UFC scrape, enrich, retrain, and promote pipeline"
) as dag:

    def scrape_new_events(**context):
        import sys
        sys.path.insert(0, "/home/mlops/octagon-analytics/scraper")
        import asyncio
        from scraper import get_existing_events, get_events, get_db

        async def run():
            conn = get_db()
            existing = get_existing_events(conn)
            conn.close()

            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                all_events = await get_events(page)
                await browser.close()

            new_events = [e for e in all_events if e["event_id"] not in existing]
            print(f"Found {len(new_events)} new events")
            context["ti"].xcom_push(key="new_event_count", value=len(new_events))
            context["ti"].xcom_push(key="new_events", value=new_events)

        asyncio.run(run())

    def branch_on_new_data(**context):
        count = context["ti"].xcom_pull(task_ids="scrape_new_events", key="new_event_count")
        return "scrape_new_fights" if count and count > 0 else "skip_run"

    def scrape_new_fights(**context):
        import sys
        sys.path.insert(0, "/home/mlops/octagon-analytics/scraper")
        import asyncio
        from scraper import get_db, get_fights, save_event, save_fighter, save_fight
        from playwright.async_api import async_playwright

        async def run():
            new_events = context["ti"].xcom_pull(task_ids="scrape_new_events", key="new_events")
            conn = get_db()

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                for event in new_events:
                    save_event(conn, event)
                    try:
                        fights = await get_fights(event["url"], page)
                        for fight in fights:
                            if not fight["fight_id"]:
                                continue
                            save_fighter(conn, fight["fighter_1"])
                            save_fighter(conn, fight["fighter_2"])
                            save_fight(conn, fight, event["event_id"])
                        print(f"Saved {len(fights)} fights for {event['name']}")
                    except Exception as e:
                        print(f"Error scraping {event['name']}: {e}")

                    import asyncio as aio
                    await aio.sleep(2)

                await browser.close()
            conn.close()

        asyncio.run(run())

    def enrich_new_fighters(**context):
        import sys
        sys.path.insert(0, "/home/mlops/octagon-analytics/scraper")
        import asyncio
        from enrich_fighters import get_db, get_unenriched_fighters, parse_fighter_stats, update_fighter
        from playwright.async_api import async_playwright

        async def run():
            conn = get_db()
            fighters = get_unenriched_fighters(conn)
            print(f"Enriching {len(fighters)} new fighters...")

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                for i, (fighter_id, name, url) in enumerate(fighters):
                    try:
                        await page.goto(url, timeout=60000)
                        await page.wait_for_selector("li.b-list__box-list-item", timeout=15000)
                        html = await page.content()
                        stats = parse_fighter_stats(html)
                        update_fighter(conn, fighter_id, stats)
                        print(f"[{i+1}/{len(fighters)}] {name} enriched")
                    except Exception as e:
                        print(f"[{i+1}/{len(fighters)}] {name} error: {e}")

                    import asyncio as aio
                    await aio.sleep(1)

                await browser.close()
            conn.close()

        asyncio.run(run())

    def retrain_model(**context):
        import subprocess
        import re

        env = os.environ.copy()
        env["PYTHONPATH"] = "/home/mlops/octagon-analytics/training/venv/lib/python3.10/site-packages"
        env["VIRTUAL_ENV"] = "/home/mlops/octagon-analytics/training/venv"

        result = subprocess.run([
            "/home/mlops/octagon-analytics/training/venv/bin/python",
            "/home/mlops/octagon-analytics/training/train.py"
        ], capture_output=True, text=True, env=env)

        print(result.stdout)
        if result.returncode != 0:
            raise Exception(f"Training failed: {result.stderr}")

        match = re.search(r"new_version:(\d+)", result.stdout)
        if match:
            context["ti"].xcom_push(key="new_version", value=int(match.group(1)))

    def evaluate_and_promote(**context):
        import mlflow
        from mlflow.tracking import MlflowClient
        from datetime import datetime

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient(MLFLOW_URI)

        new_version = context["ti"].xcom_pull(task_ids="retrain_model", key="new_version")
        new_event_count = context["ti"].xcom_pull(task_ids="scrape_new_events", key="new_event_count")

        new_model = client.get_model_version("ufc_fight_predictor", new_version)
        new_run = client.get_run(new_model.run_id)
        new_auc = new_run.data.metrics.get("test_auc", 0)

        try:
            prod_model_version = client.get_model_version_by_alias(
                "ufc_fight_predictor", "production"
            )
            prod_run = client.get_run(prod_model_version.run_id)
            prod_auc = prod_run.data.metrics.get("test_auc", 0)
            prod_version = prod_model_version.version
        except:
            prod_auc = 0
            prod_version = "none"



        threshold = 0.001
        if new_auc > prod_auc + threshold:
            client.set_registered_model_alias(
                name="ufc_fight_predictor",
                alias="production",
                version=new_version
            )
            decision = f"PROMOTED v{new_version} to Production"
            promoted = True
        else:
            decision = f"Kept existing Production v{prod_version}"
            promoted = False

        report = f"""
UFC Weekly Pipeline Report — {datetime.now().strftime('%B %d, %Y')}
================================================
New events scraped:  {new_event_count}
New model version:   v{new_version}
New AUC:             {new_auc:.4f}
Production AUC:      {prod_auc:.4f}
Improvement:         {new_auc - prod_auc:+.4f}
Decision:            {decision}
        """
        print(report)
        context["ti"].xcom_push(key="report", value=report)
        context["ti"].xcom_push(key="promoted", value=promoted)

    def send_notification(**context):
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        report = context["ti"].xcom_pull(task_ids="evaluate_and_promote", key="report")
        promoted = context["ti"].xcom_pull(task_ids="evaluate_and_promote", key="promoted")

        subject = f"UFC Pipeline Report — {'Model Promoted ✓' if promoted else 'No Change'}"

        # Configure with your email details
        sender = "loganjrivera@gmail.com"
        receiver = "loganjrivera@gmail.com"
        password = "hfzb ieoz ewis tavn"

        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = receiver
        msg["Subject"] = subject
        msg.attach(MIMEText(report, "plain"))

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender, password)
                server.sendmail(sender, receiver, msg.as_string())
            print("Email sent successfully")
        except Exception as e:
            print(f"Email failed: {e}")

    def score_pending_predictions(**context):
        import sys
        sys.path.insert(0, "/home/mlops/octagon-analytics/scraper")
        from upcoming import get_db, score_pending_predictions as _score
        conn = get_db()
        scored = _score(conn)
        conn.close()
        context["ti"].xcom_push(key="scored_count", value=scored)

    def scrape_upcoming_and_predict(**context):
        import sys
        sys.path.insert(0, "/home/mlops/octagon-analytics/scraper")
        import asyncio
        from upcoming import (get_db, get_existing_predictions,
                              get_upcoming_events, get_upcoming_fights,
                              save_upcoming_event, predict_and_store)
        from playwright.async_api import async_playwright

        async def run():
            conn = get_db()
            existing = get_existing_predictions(conn)
            total = 0

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                events = await get_upcoming_events(page)
                print(f"Found {len(events)} upcoming events")

                for event in events:
                    save_upcoming_event(conn, event)
                    try:
                        fights = await get_upcoming_fights(event["url"], page)
                        for fight in fights:
                            if predict_and_store(conn, fight, event["event_id"], existing):
                                total += 1
                    except Exception as e:
                        print(f"Error on {event['name']}: {e}")

                    import asyncio as aio
                    await aio.sleep(2)

                await browser.close()
            conn.close()
            print(f"Total new predictions stored: {total}")

        asyncio.run(run())

    def skip_run(**context):
        print("No new fights this week. Skipping retrain.")

    # Define tasks
    t_scrape = PythonOperator(
        task_id="scrape_new_events",
        python_callable=scrape_new_events,
    )

    t_branch = BranchPythonOperator(
        task_id="branch_on_new_data",
        python_callable=branch_on_new_data,
    )

    t_fights = PythonOperator(
        task_id="scrape_new_fights",
        python_callable=scrape_new_fights,
    )

    t_enrich = PythonOperator(
        task_id="enrich_new_fighters",
        python_callable=enrich_new_fighters,
    )

    t_retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
    )

    t_promote = PythonOperator(
        task_id="evaluate_and_promote",
        python_callable=evaluate_and_promote,
    )

    t_notify = PythonOperator(
        task_id="send_notification",
        python_callable=send_notification,
    )

    t_skip = PythonOperator(
        task_id="skip_run",
        python_callable=skip_run,
    )

    t_score = PythonOperator(
        task_id="score_pending_predictions",
        python_callable=score_pending_predictions,
    )

    t_upcoming = PythonOperator(
        task_id="scrape_upcoming_and_predict",
        python_callable=scrape_upcoming_and_predict,
    )



    # Wire up the DAG
    t_score >> t_scrape >> t_branch >> [t_fights, t_skip]
    t_fights >> t_enrich >> t_retrain >> t_promote >> t_notify >> t_upcoming
    t_skip >> t_upcoming
