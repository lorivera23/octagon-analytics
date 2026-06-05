#!/bin/bash
echo "Rebuilding Octagon Analytics containers..."

cd ~/octagon-analytics/docker

docker build -f Dockerfile.api -t octagon-api:latest .. && echo "API built"
docker build -f Dockerfile.mlflow -t octagon-mlflow:latest .. && echo "MLflow built"
docker build -f Dockerfile.airflow -t octagon-airflow:latest .. && echo "Airflow built"

echo "Restarting services..."
sudo systemctl restart octagon-api mlflow airflow-webserver airflow-scheduler

echo "Done. Running containers:"
docker ps
