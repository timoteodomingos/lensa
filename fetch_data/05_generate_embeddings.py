import asyncio
import os

import duckdb
import requests
from dotenv import load_dotenv
from duckdb import DuckDBPyConnection

load_dotenv()

OPEN_ROUTER_KEY = os.getenv("OPEN_ROUTER_KEY")
MODEL = "qwen/qwen3-embedding-8b"


def setup_db(con: DuckDBPyConnection) -> None:
    con.sql("""
        CREATE TABLE IF NOT EXISTS company_embeddings (
        id INTEGER,
        embedding FLOAT[4096]
        );
    """)


def load_batch(con: DuckDBPyConnection) -> list:
    return (
        con.sql("""
            SELECT id, short_summary, detailed_summary
            FROM company_summaries
            WHERE not_related_flag = false
            AND short_summary IS NOT NULL
            AND detailed_summary IS NOT NULL
            AND id NOT IN (SELECT id FROM company_embeddings)
            ORDER BY random()
            LIMIT 300
        """)
        .df()
        .to_dict(orient="records")
    )


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    response = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPEN_ROUTER_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "input": texts,
        },
    )
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]


def insert_embeddings(con: DuckDBPyConnection, rows: list) -> None:
    if not rows:
        return
    con.executemany(
        "INSERT INTO company_embeddings (id, embedding) VALUES (?, ?)",
        rows,
    )


def main():
    con = duckdb.connect("db/lensa.db")
    setup_db(con)

    try:
        while True:
            batch = load_batch(con)
            if not batch:
                print("done")
                break

            texts = [f"{r['short_summary']} {r['detailed_summary']}" for r in batch]

            try:
                embeddings = generate_embeddings(texts)
            except Exception as e:
                print(f"embedding request failed: {e}")
                continue

            rows = [(r["id"], emb) for r, emb in zip(batch, embeddings)]
            insert_embeddings(con, rows)
            print(f"inserted {len(rows)} embeddings")

    except KeyboardInterrupt:
        print("shutting down...")


if __name__ == "__main__":
    main()
