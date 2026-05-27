import asyncio
import os

import duckdb
from dotenv import load_dotenv
from duckdb import DuckDBPyConnection
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

load_dotenv()

OPEN_ROUTER_KEY = os.getenv("OPEN_ROUTER_KEY")


class CompanySummary(BaseModel):
    not_related_flag: bool
    short_summary: str | None
    detailed_summary: str | None


def setup_db(con: DuckDBPyConnection) -> None:
    con.sql("""
        CREATE TABLE IF NOT EXISTS company_summaries (
            id INTEGER,
            not_related_flag BOOLEAN,
            short_summary VARCHAR,
            detailed_summary VARCHAR
        );
    """)


def load_batch(con: DuckDBPyConnection) -> list:
    return (
        con.sql("""
            SELECT wc.id, b.name, b.activity, wc.home_content, wc.about_content
            FROM website_content wc
            JOIN brreg_data b ON wc.id = b.id
            WHERE home_content IS NOT NULL
            AND wc.id NOT IN (SELECT id FROM company_summaries)
            ORDER BY random()
            LIMIT 20
        """)
        .df()
        .to_dict(orient="records")
    )


async def process_record(agent: Agent, record: dict) -> dict | None:
    try:
        summary = await asyncio.wait_for(
            generate_summary(
                agent=agent,
                company_name=record["name"],
                company_activity=record["activity"],
                home_content=record["home_content"],
                about_content=record["about_content"]
                if isinstance(record["about_content"], str)
                else None,
            ),
            timeout=30.0,
        )
        return {
            "id": record["id"],
            "not_related_flag": summary.not_related_flag,
            "short_summary": summary.short_summary,
            "detailed_summary": summary.detailed_summary,
        }
    except asyncio.TimeoutError:
        print(f"timeout for {record['id']}, skipping")
        return None
    except Exception as e:
        print(f"failed for {record['id']}: {e}")
        return None


async def generate_summary(
    agent: Agent,
    company_name: str,
    company_activity: str,
    home_content: str,
    about_content: str | None,
) -> CompanySummary:
    content = f"Home page: {home_content}"
    if about_content:
        content += f" \n About page: {about_content}"

    prompt = f"""
    You are given website content from a Norwegian company. Your job is to create 2 summaries from the scraped company website content:

    - A semantically dense, one sentence summary that best describes the company.
    - And a more detailed covering what the company does, its unique points, target market, and any notable characteristics. ~200 words.

    Since the websites are guessed by an algorithm, it could be that the website content contains noise or is COMPLETELY UNRELATED to the company.

    Below here you'll find the companies' name and registered activity, which you can use to decide if you want to flag the content as unrelated.

    So if the page contains only noise or seems completely unrelated to the registered activity, flag it. Only flag completely unrelated, somewhat related can still be correct.
    If flagged as unrelated, set short_summary and detailed_summary to null.

    Always write your summaries in English

    COMPANY DETAILS:
    Company name: {company_name}
    Registered activity: {company_activity}

    WEBSITE CONTENT:
    {content}
    """

    response = await agent.run(prompt)
    return response.output


def insert_summaries(con: DuckDBPyConnection, results: list) -> None:
    if not results:
        return
    rows = [
        (r["id"], r["not_related_flag"], r["short_summary"], r["detailed_summary"])
        for r in results
    ]
    con.executemany(
        """
        INSERT INTO company_summaries (id, not_related_flag, short_summary, detailed_summary)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )


async def main():
    model = OpenRouterModel(
        "google/gemma-4-26b-a4b-it",
        provider=OpenRouterProvider(api_key=OPEN_ROUTER_KEY),
    )
    agent = Agent(model, output_type=CompanySummary)

    con = duckdb.connect("db/lensa.db")
    setup_db(con)

    while True:
        batch = load_batch(con)
        if not batch:
            print("script done")
            break

        results = await asyncio.gather(*[process_record(agent, r) for r in batch])
        results = [r for r in results if r is not None]
        insert_summaries(con, results)
        print(f"batch done, inserted {len(results)} summaries")


if __name__ == "__main__":
    asyncio.run(main())
