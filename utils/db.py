"""
utils/db.py
-----------
SQL data-access layer for ShopEase customer data.

Earlier version of this project read `data/customers.csv` straight into a
pandas DataFrame and filtered it in Python. This module replaces that with
a real SQLite database (data/shopease.db) and parameterized SQL queries,
while still returning plain dicts / DataFrames so the rest of the app
(crm_agent.py, app.py) doesn't need to change how it consumes the data.

The database is created and seeded automatically (from customers.csv) the
first time it's needed, using the schema in data/schema.sql.
"""

import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "shopease.db")
CSV_PATH = os.path.join(BASE_DIR, "data", "customers.csv")
SCHEMA_PATH = os.path.join(BASE_DIR, "data", "schema.sql")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    order_id      TEXT,
    order_status  TEXT,
    order_item    TEXT,
    amount        REAL,
    complaints    INTEGER DEFAULT 0,
    member_since  TEXT,
    tier          TEXT DEFAULT 'regular'
);
CREATE INDEX IF NOT EXISTS idx_customers_tier ON customers (tier);
"""


def get_connection() -> sqlite3.Connection:
    """Open a new SQLite connection to the ShopEase database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db(force_reseed: bool = False) -> None:
    """
    Create the `customers` table if it doesn't exist, and seed it from
    customers.csv the first time (or whenever force_reseed=True / the
    table is empty). Safe to call repeatedly.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()

        # Prefer the schema.sql file if present (single source of truth),
        # fall back to the inline copy so the app still works if that
        # file is ever missing.
        if os.path.exists(SCHEMA_PATH):
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                cur.executescript(f.read())
        else:
            cur.executescript(_SCHEMA_SQL)
        conn.commit()

        cur.execute("SELECT COUNT(*) FROM customers")
        count = cur.fetchone()[0]

        if (count == 0 or force_reseed) and os.path.exists(CSV_PATH):
            if force_reseed:
                cur.execute("DELETE FROM customers")

            df = pd.read_csv(CSV_PATH)
            df["customer_id"] = df["customer_id"].astype(str).str.strip()

            rows = list(
                df[
                    [
                        "customer_id", "name", "order_id", "order_status",
                        "order_item", "amount", "complaints",
                        "member_since", "tier",
                    ]
                ].itertuples(index=False, name=None)
            )
            cur.executemany(
                """
                INSERT OR REPLACE INTO customers
                    (customer_id, name, order_id, order_status, order_item,
                     amount, complaints, member_since, tier)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
    finally:
        conn.close()


def get_customer_by_id(customer_id: str) -> dict:
    """
    Fetch a single customer via a parameterized SQL SELECT.
    Returns {} if no matching customer_id is found.
    """
    init_db()
    customer_id = (customer_id or "").strip()

    conn = get_connection()
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM customers WHERE TRIM(customer_id) = ? LIMIT 1",
            (customer_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


def get_all_customers() -> pd.DataFrame:
    """Fetch every customer via SQL, returned as a DataFrame (for the UI)."""
    init_db()
    conn = get_connection()
    try:
        return pd.read_sql_query("SELECT * FROM customers ORDER BY name ASC", conn)
    finally:
        conn.close()


def search_customers(name: str = None, tier: str = None) -> pd.DataFrame:
    """
    Search customers by (partial) name and/or tier using SQL WHERE/LIKE
    filtering, instead of pandas boolean masking.
    """
    init_db()
    conn = get_connection()
    try:
        query = "SELECT * FROM customers WHERE 1=1"
        params = []
        if name:
            query += " AND name LIKE ?"
            params.append(f"%{name}%")
        if tier:
            query += " AND tier = ?"
            params.append(tier)
        query += " ORDER BY name ASC"
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()


def increment_complaints(customer_id: str) -> None:
    """Increment a customer's complaint count via SQL UPDATE."""
    init_db()
    customer_id = (customer_id or "").strip()

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE customers
               SET complaints = COALESCE(complaints, 0) + 1
             WHERE TRIM(customer_id) = ?
            """,
            (customer_id,),
        )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    # Quick manual check: `python -m utils.db` rebuilds the DB from the CSV
    # and prints what's in it.
    init_db(force_reseed=True)
    print(get_all_customers())
