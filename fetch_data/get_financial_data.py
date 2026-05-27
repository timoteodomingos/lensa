import asyncio
import os

import duckdb
import httpx
from duckdb import DuckDBPyConnection

RATE_LIMIT = 5  # requests per second
BASE_URL = "https://data.brreg.no/regnskapsregisteret/regnskap"


def setup_db(con: DuckDBPyConnection) -> None:
    con.sql("""
        CREATE TABLE IF NOT EXISTS company_financials (
        id INTEGER,
        year INTEGER,
        revenue BIGINT,
        operating_profit BIGINT,
        net_profit BIGINT,
        total_assets BIGINT,
        equity BIGINT,
        total_debt BIGINT,
        payroll_cost BIGINT,
        PRIMARY KEY (id, year)
        );
    """)


def load_batch(con: DuckDBPyConnection) -> list:
    return (
        con.sql("""
            SELECT DISTINCT w.id, w.url
            FROM websites w
            JOIN brreg_data b ON w.id = b.id
            WHERE w.website_found = true
            AND w.id NOT IN (SELECT DISTINCT id FROM company_financials)
            ORDER BY random()
            LIMIT 100
        """)
        .df()
        .to_dict(orient="records")
    )


def parse_financials(data: list, org_id: int) -> list:
    rows = []
    for entry in data:
        try:
            period = entry.get("regnskapsperiode", {})
            year = int(period.get("tilDato", "")[:4])

            res = entry.get("resultatregnskapResultat", {})
            drift = res.get("driftsresultat", {})
            inntekter = drift.get("driftsinntekter", {})
            kostnad = drift.get("driftskostnad", {})
            eiendeler = entry.get("eiendeler", {})
            eg = entry.get("egenkapitalGjeld", {})

            rows.append(
                (
                    org_id,
                    year,
                    inntekter.get("sumDriftsinntekter"),
                    drift.get("driftsresultat"),
                    res.get("aarsresultat"),
                    eiendeler.get("sumEiendeler"),
                    eg.get("egenkapital", {}).get("sumEgenkapital"),
                    eg.get("gjeldOversikt", {}).get("sumGjeld"),
                    kostnad.get("loennskostnad"),
                )
            )
        except Exception:
            continue
    return rows


async def fetch_financials(client: httpx.AsyncClient, org_id: int) -> list:
    try:
        response = await client.get(f"{BASE_URL}/{org_id}", timeout=15)
        if response.status_code != 200:
            return []
        data = response.json()
        if not isinstance(data, list):
            return []
        return parse_financials(data, org_id)
    except Exception as e:
        print(f"failed for {org_id}: {e}")
        return []


def insert_financials(con: DuckDBPyConnection, rows: list) -> None:
    if not rows:
        return
    con.executemany(
        """
        INSERT INTO company_financials
        (id, year, revenue, operating_profit, net_profit,
        total_assets, equity, total_debt, payroll_cost)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        rows,
    )


async def main():
    con = duckdb.connect("db/lensa.db")
    setup_db(con)

    async with httpx.AsyncClient() as client:
        while True:
            batch = load_batch(con)
            if not batch:
                print("done")
                break

            all_rows = []
            for i, company in enumerate(batch):
                task = asyncio.create_task(fetch_financials(client, company["id"]))
                task.add_done_callback(lambda t: all_rows.extend(t.result() or []))
                if (i + 1) % RATE_LIMIT == 0:
                    await asyncio.sleep(1)

            await asyncio.sleep(1)  # let last tasks finish
            insert_financials(con, all_rows)
            print(f"batch done, inserted {len(all_rows)} rows")


if __name__ == "__main__":
    asyncio.run(main())
