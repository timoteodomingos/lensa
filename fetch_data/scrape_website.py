import asyncio
import json
import os
from datetime import timedelta
from urllib.parse import urlparse

import duckdb
from crawlee import ConcurrencySettings, Request
from crawlee.crawlers import (
    PlaywrightCrawler,
    PlaywrightCrawlingContext,
)
from dotenv import load_dotenv
from duckdb import DuckDBPyConnection
from openrouter import OpenRouter
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider


class AboutPage(BaseModel):
    about_page_found: bool
    about_page_url: str | None


load_dotenv()
OPEN_ROUTER_KEY = os.getenv("OPEN_ROUTER_KEY")


def load_batch(con: DuckDBPyConnection) -> list:
    print("loading batch...")
    batch = (
        con.sql(
            """
        WITH do_not_retry AS (
            SELECT id FROM log_website_scrape_failed
            GROUP BY id
            HAVING COUNT(id) >= 3
            UNION ALL
            SELECT id from website_content
        )
        SELECT b.name, w.id, w.url, b.activity FROM websites w
        JOIN brreg_data b ON w.id = b.id
        WHERE w.url IS NOT NULL
        AND w.confidence = 'high'
        AND w.id NOT IN (SELECT id FROM do_not_retry)
        ORDER BY random()
        LIMIT 100
        """
        )
        .df()
        .to_dict(orient="records")
    )

    return batch


def setup_db(con: DuckDBPyConnection) -> None:
    con.sql(
        """
        CREATE TABLE IF NOT EXISTS website_content (
        id INTEGER,
        base_url VARCHAR,
        about_url VARCHAR,
        home_content VARCHAR,
        about_content VARCHAR
        );
        """
    )

    con.sql(
        """
        CREATE TABLE IF NOT EXISTS log_website_scrape_failed (
        id INTEGER,
        base_url VARCHAR,
        error VARCHAR
        );
        """
    )


async def find_about_page(
    agent: Agent, links: list, company_name: str, company_activity: str
) -> AboutPage:
    prompt = f"""
    You are given a list of URLs from the homepage of a Norwegian company.

    Your task is to identify whether an "about" page exists among these links.

    Company name: {company_name}
    Registered activity: {company_activity}

    Look for pages that clearly describe the company — typically named "about", "om oss", "company", "selskap", "hvem er vi", or similar Norwegian/English equivalents.

    Rules:
    - Only return a URL that is explicitly present in the list
    - Do not guess or infer URLs that are not listed
    - Only return a URL if you are confident it is an about page — do not include links that are only vaguely plausible
    - If no clear about page exists, return about_page_found: false and about_page_url: null
    - Return the URL in its full form as it appears in the list

    Links:
    {links}
    """

    response = await agent.run(prompt)
    return response.output


def insert_results(con: DuckDBPyConnection, results: list) -> None:
    if not results:
        return
    rows = [
        (
            r["id"],
            r["base_url"],
            r["about_url"],
            r["home_content"],
            r["about_content"],
        )
        for r in results
    ]
    con.executemany(
        """
        INSERT INTO website_content (id, base_url, about_url, home_content, about_content)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )


def insert_failed(con: DuckDBPyConnection, failed: list) -> None:
    if not failed:
        return
    rows = [(r["id"], r["url"], r["error"]) for r in failed]
    con.executemany(
        """
        INSERT INTO log_website_scrape_failed (id, base_url, error)
        VALUES (?, ?, ?)
        """,
        rows,
    )


async def main():

    model = OpenRouterModel(
        "google/gemma-4-26b-a4b-it",
        provider=OpenRouterProvider(api_key=OPEN_ROUTER_KEY),
    )
    agent = Agent(model, output_type=AboutPage)
    crawler = PlaywrightCrawler(
        concurrency_settings=ConcurrencySettings(max_concurrency=20),
        max_request_retries=1,
        request_handler_timeout=timedelta(seconds=30),
    )
    con = duckdb.connect("db/lensa.db")
    setup_db(con)

    results = []
    failed = []

    @crawler.router.default_handler
    async def handler(context: PlaywrightCrawlingContext):
        data = context.request.user_data
        base = urlparse(context.request.url).netloc.replace("www.", "")
        company_id = data["id"]
        company_name = data["name"]
        company_activity = data["activity"]

        home_content_full = await context.page.evaluate("() => document.body.innerText")
        home_content = home_content_full[:10000]
        url = context.request.url
        all_links = await context.page.eval_on_selector_all(
            # "nav a[href]", "els => els.map(el => el.href)"
            "a[href]",
            "els => els.map(el => el.href)",
        )

        filtered_links = [
            l
            for l in dict.fromkeys(all_links)
            if base in urlparse(l).netloc.replace("www.", "")
            and len(urlparse(l).path.strip("/").split("/")) <= 1
        ][:100]

        about = None

        try:
            about = await find_about_page(
                agent=agent,
                company_name=company_name,
                company_activity=company_activity,
                links=filtered_links,
            )
        except Exception as e:
            print(f"LLM call for about page failed for {url}: {e}")

        about_content = None

        if about and about.about_page_found and about.about_page_url:
            try:
                about_url = about.about_page_url
                await context.page.goto(about_url)
                about_content_full = await context.page.evaluate(
                    "() => document.body.innerText"
                )
                about_content = about_content_full[:10000]
            except Exception as e:
                print(f"failed to scrape about page: {e}")

        results.append(
            {
                "id": data["id"],
                "base_url": context.request.url,
                "home_content": home_content,
                "about_page_found": about.about_page_found if about else None,
                "about_url": about.about_page_url if about else None,
                "about_content": about_content,
            }
        )

    @crawler.failed_request_handler
    async def failed_handler(context: PlaywrightCrawlingContext, error: Exception):
        failed.append(
            {
                "id": context.request.user_data.get("id"),
                "url": context.request.url,
                "error": str(error),
            }
        )

    try:
        while True:
            batch = load_batch(con)
            if not batch:
                print("no more batches, done")
                break

            urls = [
                Request.from_url(
                    f"https://{w['url']}",
                    user_data={
                        "id": w["id"],
                        "name": w["name"],
                        "activity": w["activity"],
                    },
                )
                for w in batch
            ]
            await crawler.run(urls)
            insert_results(con, results)
            insert_failed(con, failed)
            print(f"batch done, inserted {len(results)} results")
            results.clear()
            failed.clear()
    except KeyboardInterrupt:
        print("shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
