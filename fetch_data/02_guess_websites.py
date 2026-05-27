import asyncio
import json
from typing import Literal, Optional

import duckdb
import ollama
import pandas
from ddgs import DDGS
from duckdb import DuckDBPyConnection
from ollama import AsyncClient
from pydantic import BaseModel


class CompanySite(BaseModel):
    site_found: bool
    confidence: Optional[Literal["low", "medium", "high"]] = None
    url: Optional[str] = None


def setup_db(con: DuckDBPyConnection) -> None:
    con.sql(
        """
        CREATE TABLE IF NOT EXISTS websites (
            id INTEGER,
            website_found BOOLEAN,
            url VARCHAR,
            confidence VARCHAR
        );
        """
    )

    con.sql(
        """
        CREATE TABLE IF NOT EXISTS log_search_failed (
            id INTEGER,
            exception VARCHAR,
            time_inserted TIMESTAMP DEFAULT now()
        );
        """
    )


def load_batch(con: DuckDBPyConnection) -> list:
    print("loading batch...")
    batch = (
        con.sql(
            """
        WITH do_not_retry AS (
            SELECT id FROM log_search_failed
            GROUP BY id
            HAVING COUNT(id) >= 3

            UNION ALL

            SELECT id from websites
        )

        SELECT
            id,
            name,
            address,
            city,
            activity
        FROM brreg_data
        WHERE
            LOWER(org_type) = 'as'
            AND LOWER(city) = 'oslo'
            AND id NOT IN (SELECT id FROM do_not_retry)
            AND (name IS NOT NULL or name != '')
            AND (address IS NOT NULL or address != '')
            AND (city IS NOT NULL or city != '')
            AND (activity IS NOT NULL or activity != '')
        ORDER BY random()
        LIMIT 50
        """
        )
        .df()
        .to_dict(orient="records")
    )
    return batch


async def find_company_website(
    name: str, address: str, city: str, activity: str, ollama_client: AsyncClient
) -> tuple:
    search_results = await asyncio.to_thread(search_company_info, name, address, city)
    if not search_results:
        return ()

    print(search_results)
    website = await guess_website(
        ollama_client, search_results, name, address, city, activity
    )
    if not website:
        return ()

    return website


def search_company_info(company_name: str, address: str, city: str) -> list:
    search_query = f"{company_name} {address} {city}"
    with DDGS() as ddgs:
        results = [
            r
            for r in ddgs.text(
                search_query,
                max_results=None,
                backend="brave,yahoo, yandex",
            )[:20]
        ]
        return results or []


async def guess_website(
    ollama_client: AsyncClient,
    search_results: str,
    name: str,
    address: str,
    city: str,
    activity: str,
) -> dict:
    prompt = generate_prompt(name, address, activity, search_results)

    response = await ollama_client.chat(
        model="gemma4:26b",
        messages=[{"role": "user", "content": prompt}],
        think=False,
        stream=False,
        format=CompanySite.model_json_schema(),
        options={"temperature": 0},
    )
    if response:
        return json.loads(response.message.content)


def generate_prompt(name: str, address: str, activity: str, search_results: str) -> str:
    prompt = f"""
    We are looking for which website from the search results fits the company from the norwegian chamber of commerce.
    We used the companies name and address for the search query. Your job is to return the website which belongs to the company we are looking for.

    If you find the site, make sure to fill in the confidence level and url.

    Sometimes there exists no website. If none of the search results looks plausible to be the website, return false for site found, and leave confidence and url empty.

    If the company appears to be a holding/shell company with no independent web presence, prefer returning null over attributing a parent company's website.

    Below we'll provide you with the company name and activity (activity is sometimes useful to spot if the description is completely unrelated, then it might not be the correct site). Its exclusively norwegian companies, so beware that the result will usually be .no,
    but international domains are not uncommon. Just be extra wary if the name is international, check if it likely belongs to the company or not.

    There are 20 search results, but usually if the site is found, it should be one of the first results.

    For your json response, make sure to follow the following instructions for the url:

        - without www. or https prefix. Just like so: example.com
        so no slashes or subdirectories like example.com/example_page/example or whatever. Add the site dry as can be, just the root domain.

    COMPANY NAME: {name}
    COMPANY ADDRESS: {address}
    COMPANY ACTIVITY: {activity}


    Confidence level:
        High — name matches exactly, domain is clearly theirs
        Medium — likely match but name or activity is only partially confirmed
        Low — plausible but uncertain

    SEARCH RESULTS:

    Be careful to not hallucinate any websites which are not inside any of the HREFS

    Make sure, when a website is found, to always fill in confidence level and url.

    If no website is found, set bool to false and return None for confidence level and URL.



    {search_results}
    """
    return prompt


def insert_failed(con: DuckDBPyConnection, failed: list) -> None:
    if not failed:
        return
    con.executemany(
        "INSERT INTO log_search_failed (id, exception) VALUES (?, ?)",
        failed,
    )


def insert_success(con: DuckDBPyConnection, success: list) -> None:
    if not success:
        return
    rows = [
        (id, result["site_found"], result.get("url"), result.get("confidence"))
        for id, result in success
    ]
    con.executemany(
        """
        INSERT INTO websites (id, website_found, url, confidence)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )


async def main():
    ollama_client = AsyncClient()

    con = duckdb.connect("db/lensa.db")

    setup_db(con)

    while True:
        batch = load_batch(con)

        if not batch:
            print("finished all")
            break

        print(f"starting batch of {len(batch)}...")
        tasks = []
        ids = [company["id"] for company in batch]

        for company in batch:
            task = asyncio.create_task(
                find_company_website(
                    company["name"],
                    company["address"],
                    company["city"],
                    company["activity"],
                    ollama_client,
                )
            )
            tasks.append(task)
            await asyncio.sleep(1.1)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        results_list = list(zip(ids, results))
        failed = [
            (id, str(result))
            for id, result in results_list
            if result is None or isinstance(result, Exception)
        ]
        # in our script result is a dict i think but lets check
        success = [
            (id, result)
            for id, result in results_list
            if result and not isinstance(result, Exception)
        ]
        print(f"{len(failed)} failed, {len(success)} succeeded from batch.")
        insert_failed(con, failed)
        insert_success(con, success)


if __name__ == "__main__":
    asyncio.run(main())
