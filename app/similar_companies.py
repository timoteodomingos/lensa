import os

import duckdb
import requests
from dotenv import load_dotenv
from duckdb import DuckDBPyConnection

load_dotenv()

OPEN_ROUTER_KEY = os.getenv("OPEN_ROUTER_KEY")
MODEL = "qwen/qwen3-embedding-8b"

search_query = "Instruct: Given a search query, retrieve relevant company descriptions\nQuery: speciality coffee fairtrade"

con = duckdb.connect("db/lensa.db")

response = requests.post(
    "https://openrouter.ai/api/v1/embeddings",
    headers={
        "Authorization": f"Bearer {OPEN_ROUTER_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": MODEL,
        "input": search_query,
    },
)
response.raise_for_status()
data = response.json()
embedding = data["data"][0]["embedding"]
# print(embedding)

top = con.execute(
    """
    SELECT
        e.id,
        b.name,
        cs.short_summary,
        array_cosine_distance(e.embedding, ?::FLOAT[4096]) as distance
    FROM company_embeddings e
    JOIN brreg_data b ON e.id = b.id
    JOIN company_summaries cs ON e.id = cs.id
    ORDER BY distance ASC
    LIMIT 15
""",
    [embedding],
).df()

print(top)
