from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import pendulum
import os

# NOTE: I eventually want to get rid of the sys.insert lines and move to setting PYTHONPATH in the Dockerfile
#       Also fix the databaase connection in score predictions, the function already manages connection

MLFLOW_URI = os.environ["MLFLOW_URI"]

default_args = {
    "owner": "mlops",
    "retries": 0,
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

    def run_incremental_task(**context):
        import sys
        sys.path.insert(0, "/home/mlops/octagon-analytics/scraper")
        import asyncio
        from scraper import run_incremental
        return asyncio.run(run_incremental())
    
    def score_pending_predictions(**context):
        import sys
        sys.path.insert(0, "/home/mlops/octagon-analytics/scraper") # for getting the db connection
        sys.path.insert(0, "/home/mlops/octagon-analytics/pipeline_utils")
        from db import get_db
        from score_predictions import score_pending_predictions as _score
        from contextlib import closing
        with closing(get_db()) as conn:
            scored = _score(conn)
        context["ti"].xcom_push(key="scored_count", value=scored)

    def branch_on_retrain(**context):
        count = context["ti"].xcom_pull(task_ids="run_incremental")  
        return "retrain_model" if count and count > 0 else "skip_retrain"


    def enrich_new_fighters_task(**context):
        import sys
        sys.path.insert(0, "/home/mlops/octagon-analytics/scraper")
        import asyncio
        from enrich_fighters import run_enrichment 
        asyncio.run(run_enrichment())

    def scrape_upcoming_and_predict_task(**context):
        import sys
        sys.path.insert(0, "/home/mlops/octagon-analytics/scraper")
        import asyncio
        from upcoming import run_upcoming           
        asyncio.run(run_upcoming())

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
        from mlflow.exceptions import MlflowException
        import sys
        sys.path.insert(0, "/home/mlops/octagon-analytics/pipeline_utils")
        from promotion_smtp import build_report

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient(MLFLOW_URI)

        new_version = context["ti"].xcom_pull(task_ids="retrain_model", key="new_version")
        new_event_count = context["ti"].xcom_pull(task_ids="run_incremental")  # return_value, fixed

        new_model = client.get_model_version("ufc_fight_predictor", new_version)
        new_run = client.get_run(new_model.run_id)
        new_auc = new_run.data.metrics.get("test_auc", 0)

        try:
            prod_mv = client.get_model_version_by_alias("ufc_fight_predictor", "production")
            prod_run = client.get_run(prod_mv.run_id)
            prod_auc = prod_run.data.metrics.get("test_auc", 0)
            prod_version = prod_mv.version
        except MlflowException as e:
            # Cold-start: no "production" alias yet - treat prod as absent so any model promotes.
            # Matched on message text because this MLflow version returns a generic
            # INVALID_PARAMETER_VALUE code (not RESOURCE_DOES_NOT_EXIST), so the code can't
            # discriminate missing-alias from other bad-param errors. Version-coupled: if MLflow
            # res the message, this check silently fails — revisit on MLflow upgrade.
            if "not found" in str(e).lower():
                prod_auc = 0
                prod_version = "none"
            else:
                raise   # MLflow unreachable / 500 / other — do NOT promote blind against prod_auc=0

        threshold = 0.001
        if new_auc > prod_auc + threshold:
            client.set_registered_model_alias("ufc_fight_predictor", "production", new_version)
            decision = f"PROMOTED v{new_version} to Production"
            promoted = True
        else:
            decision = f"Kept existing Production v{prod_version}"
            promoted = False

        report = build_report(new_event_count, new_version, new_auc, prod_auc, decision)
        print(report)
        context["ti"].xcom_push(key="report", value=report)
        context["ti"].xcom_push(key="promoted", value=promoted)

    def send_notification(**context):
        import os, sys
        sys.path.insert(0, "/home/mlops/octagon-analytics/pipeline_utils")
        from promotion_smtp import send_email

        report = context["ti"].xcom_pull(task_ids="evaluate_and_promote", key="report")
        promoted = context["ti"].xcom_pull(task_ids="evaluate_and_promote", key="promoted")
        subject = f"UFC Pipeline Report — {'Model Promoted ✓' if promoted else 'No Change'}"

        send_email(
            subject,
            report,
            sender="loganjrivera@gmail.com",
            receiver="loganjrivera@gmail.com",
            password=os.environ["GMAIL_APP_PASSWORD"],
        )


    # Define tasks
    t_scrape = PythonOperator(
        task_id="run_incremental",
        python_callable=run_incremental_task,
    )

    t_branch = BranchPythonOperator(
        task_id="branch_on_retrain",
        python_callable=branch_on_retrain,
    )


    t_enrich = PythonOperator(
        task_id="enrich_new_fighters",
        python_callable=enrich_new_fighters_task,
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

    t_skip = EmptyOperator(
        task_id="skip_retrain",
    )

    t_score = PythonOperator(
        task_id="score_pending_predictions",
        python_callable=score_pending_predictions,
    )

    t_upcoming = PythonOperator(
        task_id="scrape_upcoming_and_predict",
        python_callable=scrape_upcoming_and_predict_task,
        trigger_rule="none_failed_min_one_success",
    )



    # Wire up the DAG
    t_scrape >> [t_branch, t_enrich, t_score]
    t_branch >> [t_retrain, t_skip]
    t_retrain >> t_promote >> t_notify >> t_upcoming
    t_skip >> t_upcoming
    t_enrich >> t_upcoming

