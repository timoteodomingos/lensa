import os

import duckdb
import numpy as np
import pandas as pd
import umap
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from toponymy import KeyphraseBuilder, Toponymy, ToponymyClusterer
from toponymy.llm_wrappers import AsyncOpenAINamer

load_dotenv()
OPEN_ROUTER_KEY = os.getenv("OPEN_ROUTER_KEY")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_API_KEY"] = OPEN_ROUTER_KEY


# def reduce_embeddings(df: pd.DataFrame, n_components: int) -> np.ndarray:
#     """Reduce embeddings to `n_components` dimensions with UMAP."""
#     embeddings = np.stack(df["embedding"].values).astype(np.float32)
#     reducer = umap.UMAP(
#         n_components=n_components,
#         n_neighbors=15,
#         min_dist=0.05,
#         metric="cosine",
#         random_state=42,
#     )
#     return reducer.fit_transform(embeddings)


con = duckdb.connect("db/lensa.db")
df = con.sql("""
    SELECT
        cs.id,
        cs.detailed_summary,
        ce.embedding,
        cc.x,
        cc.y
    FROM company_summaries cs
    JOIN company_embeddings ce ON cs.id = ce.id
    JOIN company_xy_coords cc ON cs.id = cc.id
    WHERE cs.not_related_flag = FALSE
      AND cs.detailed_summary IS NOT NULL
""").df()
print(f"Loaded {len(df)} companies")

embeddings = np.stack(df["embedding"].values).astype(np.float32)
documents = df["detailed_summary"].tolist()

# print("Computing 5D UMAP for clustering...")
# clusterable_coords = np.ascontiguousarray(
#     reduce_embeddings(df, n_components=5), dtype=np.float32
# )
clusterable_coords = np.ascontiguousarray(df[["x", "y"]].values, dtype=np.float32)

embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
llm = AsyncOpenAINamer(
    api_key=OPEN_ROUTER_KEY,
    model="google/gemma-4-26b-a4b-it",
    max_concurrent_requests=20,
)
topic_model = Toponymy(
    llm_wrapper=llm,
    text_embedding_model=embedding_model,
    clusterer=ToponymyClusterer(min_clusters=4, verbose=True),
    keyphrase_builder=KeyphraseBuilder(
        ngram_range=(1, 6), max_features=15_000, verbose=True
    ),
    object_description="Norwegian company descriptions",
    corpus_description="collection of Norwegian companies",
)
np.random.seed(42)
topic_model.fit(
    documents,
    embedding_vectors=embeddings,
    clusterable_vectors=clusterable_coords,
)

n_layers = len(topic_model.cluster_layers_)
print(f"Found {n_layers} cluster layer(s)")

labels_df = pd.DataFrame({"id": df["id"].values})
for i, layer in enumerate(reversed(topic_model.cluster_layers_)):
    labels_df[f"label_layer_{i}"] = layer.topic_name_vector

con.sql("DROP TABLE IF EXISTS company_topics")
con.sql("CREATE TABLE company_topics AS SELECT * FROM labels_df")
print(f"Inserted topic labels for {len(labels_df)} companies")
con.close()
