import duckdb
import numpy as np
import pandas as pd
import umap
from duckdb import DuckDBPyConnection


def setup_db(con: DuckDBPyConnection) -> None:
    con.sql("""
        CREATE TABLE IF NOT EXISTS company_xy_coords (
        id INTEGER,
        x DOUBLE,
        y DOUBLE
        );
    """)


def load_embeddings(con: DuckDBPyConnection) -> pd.DataFrame:
    return con.sql("""
        SELECT
            id, embedding
        FROM
            company_embeddings
        WHERE id NOT IN (SELECT id FROM company_xy_coords)
    """).df()


def reduce_to_2d(df: pd.DataFrame) -> pd.DataFrame:
    embeddings = np.stack(df["embedding"].values).astype(np.float32)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.05,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)
    return pd.DataFrame(
        {
            "id": df["id"].values,
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    )


def insert_coords(con: DuckDBPyConnection, coords_df: pd.DataFrame) -> None:
    con.sql("INSERT INTO company_xy_coords SELECT * FROM coords_df")


def main() -> None:
    con = duckdb.connect("db/lensa.db")
    setup_db(con)

    df = load_embeddings(con)
    if df.empty:
        print("No new embeddings, shutting down..")
        return

    print(f"Reducing {len(df)} embeddings...")
    coords_df = reduce_to_2d(df)

    insert_coords(con, coords_df)
    print(f"Inserted {len(coords_df)} coordinate rows")


if __name__ == "__main__":
    main()
