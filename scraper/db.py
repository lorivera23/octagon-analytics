import os
import psycopg2

DB_CONFIG = {
    "dbname":   os.environ["DB_NAME"],
    "user":     os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "host":     os.environ["DB_HOST"],
    "port":     os.environ.get("DB_PORT", "5432"),
}

def get_db():
    return psycopg2.connect(**DB_CONFIG)
