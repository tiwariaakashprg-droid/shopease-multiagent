-- ShopEase customer database schema
-- Backed by SQLite (data/shopease.db). Seeded automatically from
-- data/customers.csv the first time utils/db.py runs, so this file
-- doubles as documentation and as the source of truth for the schema.

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
