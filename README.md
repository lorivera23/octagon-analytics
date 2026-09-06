# Octagon Analytics

**An end-to-end MLOps pipeline for automated UFC fight prediction.**

Octagon Analytics is a self-hosted machine learning system that collects UFC fight data, trains and evaluates an XGBoost prediction model, manages model versions with MLflow, serves predictions through FastAPI, and presents results in a React dashboard.

The project was originally deployed on a personal Proxmox homelab and designed to automate the full lifecycle from new fight data to production predictions.

> **Project status:** Archived portfolio project. The original homelab deployment and database are no longer active. The repository preserves the application, ML pipeline, orchestration, and container configuration, but is not currently hosted as a live service.

## Architecture

```text
                     ┌──────────────────┐
                     │    UFCStats      │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ Scraping Layer   │
                     │ Playwright + BS4 │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │   PostgreSQL     │
                     │ fights/fighters  │
                     │ predictions      │
                     └───────┬──────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
        ┌─────────────────┐     ┌─────────────────┐
        │ Training        │     │ Prediction      │
        │ XGBoost         │     │ Scoring         │
        │ Feature Eng.    │     │                 │
        └────────┬────────┘     └─────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ MLflow          │
        │ Tracking        │
        │ Model Registry  │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ FastAPI         │
        │ Inference API   │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ React Dashboard │
        │ Predictions     │
        │ Model Metrics   │
        └─────────────────┘

              Orchestrated by
              Apache Airflow
```

## What the System Does

The Airflow pipeline is scheduled to run weekly and coordinates the ML lifecycle:

1. **Scrape new UFC results** from UFCStats.
2. **Enrich fighter records** with available fighter information.
3. **Score previously generated predictions** once fight results become available.
4. **Detect new training data** and conditionally retrain the model.
5. **Evaluate the new model** against the current production model.
6. **Register model versions in MLflow** and promote improved models to the `production` alias.
7. **Send an automated pipeline report** containing model and promotion information.
8. **Scrape upcoming UFC events** and generate predictions through the inference API.

If no new completed events are found, unnecessary model retraining is skipped.

## Machine Learning Pipeline

### Model

The prediction model uses **XGBoost** to estimate the probability that either fighter wins a matchup.

Rather than treating fighter statistics as static observations, the training pipeline constructs features using information available **before each fight**.

Features include:

* Height difference
* Reach difference
* UFC fight experience difference
* Current win-streak difference
* Recent form over the previous five fights
* Days since each fighter's previous fight
* Age difference at fight time
* Stance matchup
* Weight class

### Preventing Temporal Leakage

A major design consideration is ensuring that historical fights are represented using information that would actually have been available at prediction time.

Fighter history is therefore calculated chronologically. The current fight result is added to a fighter's record **only after** the features for that fight have been generated.

Current aggregate career statistics are retained for display purposes but are intentionally excluded from model training because they represent present-day snapshots and could leak future information into historical training examples.

### Chronological Evaluation

Instead of randomly splitting individual fights, the dataset is divided chronologically:

```text
Older fights                       Newer fights
─────────────────────────────────────────────────► time

        Training Data              Test Data
```

The newest portion of event dates is reserved for testing.

This more closely represents the real deployment problem: **train using fights that have already happened and evaluate on fights occurring later.**

### Fighter-Order Augmentation

Training data is augmented by mirroring each matchup.

For example:

```text
Fighter A vs Fighter B
```

also becomes:

```text
Fighter B vs Fighter A
```

Directional features reverse sign and the target is inverted.

This augmentation is performed **only after the chronological split and only on the training set**, preventing mirrored versions of the same fight from leaking across training and evaluation data.

## Model Tracking and Promotion

MLflow handles experiment tracking and model versioning.

Each training run records information including:

* Training accuracy
* Test accuracy
* Test ROC AUC
* Test log loss
* Training/test row counts
* Chronological test cutoff
* XGBoost hyperparameters

Successful models are registered as versions of:

```text
ufc_fight_predictor
```

During the weekly pipeline, Airflow compares the candidate model's test AUC with the model currently assigned to the MLflow `production` alias.

A new model is promoted only when it exceeds the production model by the configured improvement threshold.

This creates a simple **human-observable model promotion workflow** rather than automatically replacing the deployed model after every training run.

## Prediction API

A **FastAPI** service loads the model assigned to MLflow's `production` alias and exposes inference and application endpoints.

For a requested matchup, the API:

1. Retrieves both fighter profiles from PostgreSQL.
2. Reconstructs fighter history as of the prediction date.
3. Builds features using the same schema used during training.
4. Runs the production XGBoost model.
5. Returns win probabilities, predicted winner, confidence, and model metadata.
6. Stores scheduled-fight predictions so they can later be scored against actual results.

Historical feature generation is performed relative to the event date, keeping training and inference behavior aligned.

## Upcoming Fight Predictions

The upcoming-event pipeline uses Playwright and BeautifulSoup to collect scheduled UFC events and fights.

Each matchup is sent to the FastAPI inference service:

```text
Upcoming Event
      │
      ▼
Scrape Matchups
      │
      ▼
FastAPI /predict
      │
      ▼
Production MLflow Model
      │
      ▼
Store Prediction
      │
      ▼
Score After Fight Occurs
```

This allows predictions to be created before an event and evaluated automatically after the corresponding results are scraped.

## Dashboard

The React/Vite frontend provides three primary views:

### Upcoming Fights

Displays model predictions for scheduled UFC matchups.

### Model Performance

Surfaces historical prediction and model-performance information exposed by the API.

### Custom Matchup

Allows two fighters to be selected for an on-demand prediction using the current production model.

The frontend was developed as a local interface for the homelab deployment rather than as a publicly hosted application.

## Technology Stack

| Layer                  | Technology                    |
| ---------------------- | ----------------------------- |
| Machine Learning       | XGBoost, scikit-learn, pandas |
| Experiment Tracking    | MLflow                        |
| Workflow Orchestration | Apache Airflow                |
| API                    | FastAPI                       |
| Database               | PostgreSQL                    |
| Web Scraping           | Playwright, BeautifulSoup     |
| Frontend               | React, Vite                   |
| Infrastructure         | Docker / Docker Compose       |
| Original Host          | Proxmox homelab               |
| Notifications          | SMTP                          |

## Repository Structure

```text
octagon-analytics/
│
├── airflow/
│   └── dags/
│       └── ufc_pipeline.py       # Weekly ML orchestration
│
├── api/
│   └── main.py                   # FastAPI inference/application API
│
├── docker/
│   ├── docker-compose.yml        # Service orchestration
│   ├── Dockerfile.airflow
│   ├── Dockerfile.api
│   ├── Dockerfile.mlflow
│   └── postgres/
│       └── init.sql
│
├── frontend/
│   └── src/                      # React dashboard
│
├── pipeline_utils/
│   ├── promotion_smtp.py         # Pipeline reports/notifications
│   └── score_predictions.py      # Scores completed predictions
│
├── scraper/
│   ├── scraper.py                # Historical fight/event scraping
│   ├── upcoming.py               # Upcoming event prediction pipeline
│   ├── enrich_fighters.py        # Fighter profile enrichment
│   └── db.py                     # Database connection utilities
│
└── training/
    └── train.py                  # Feature engineering + XGBoost training
```

## Infrastructure

The original project ran as a collection of containerized services on a self-hosted **Proxmox** server.

Docker Compose defines the major services:

```text
PostgreSQL
    ├── MLflow
    ├── Airflow Webserver
    ├── Airflow Scheduler
    └── FastAPI
```

Configuration and credentials are supplied through environment variables rather than committed secrets.

An example environment configuration is provided in:

```text
docker/.env.example
```

## Reproducibility Note

This repository is preserved primarily as a portfolio representation of the original system.

The original PostgreSQL database and historical deployment environment are no longer available, and the complete historical database schema/migration history was not preserved in the repository. As a result, the current Docker configuration documents the service architecture but should **not be considered a one-command reproduction of the original deployment from an empty database**.

The application code, feature engineering, model-training logic, orchestration workflow, inference API, frontend, and container definitions are retained.

## Key Design Lessons

This project began as a fight-prediction model but evolved into an exercise in building the infrastructure surrounding a machine learning system.

Some of the most important engineering lessons were:

* A good offline model is only one component of an ML product.
* Training and inference must construct features consistently.
* Random train/test splits can produce misleading results for temporal data.
* Feature engineering must respect what information existed at prediction time.
* Model retraining and model deployment should be separate decisions.
* Prediction quality should be measured after deployment, not only during training.
* Workflow orchestration makes data collection, retraining, evaluation, and inference reproducible.
* Model registries provide a useful boundary between experimentation and production inference.

## Project Status

**Archived / portfolio project**

Octagon Analytics was originally built and deployed on a personal homelab as an exploration of end-to-end MLOps architecture.

The live infrastructure has since been retired, but the repository is maintained as a demonstration of the system design and engineering work involved in building an automated ML pipeline from data collection through inference.
