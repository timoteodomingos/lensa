import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import umap
from dash import Dash, dcc, html

con = duckdb.connect("db/lensa.db", read_only=True)

df = con.sql("""
    SELECT
        e.id,
        e.embedding,
        b.name,
        cs.short_summary
    FROM company_embeddings e
    JOIN brreg_data b ON e.id = b.id
    JOIN company_summaries cs ON e.id = cs.id
    WHERE cs.not_related_flag = false
    AND cs.short_summary IS NOT NULL
""").df()

np.random.seed(42)
df["revenue"] = np.random.uniform(1_000_000, 500_000_000, size=len(df))
df["log_revenue"] = np.log1p(df["revenue"])

embeddings_matrix = np.array(df["embedding"].tolist())

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
coords = reducer.fit_transform(embeddings_matrix)

df["x"] = coords[:, 0]
df["y"] = coords[:, 1]
df = df.sort_values("log_revenue", ascending=False)

fig = px.scatter(
    df,
    x="x",
    y="y",
    size="revenue",
    size_max=30,
    hover_name="name",
    hover_data={"short_summary": True, "x": False, "y": False, "log_revenue": False},
    title="Norwegian Companies — Semantic Map",
    template="plotly_dark",
)
fig.update_traces(marker=dict(opacity=0.7))
fig.update_layout(showlegend=False)

app = Dash(__name__)
app.layout = html.Div(
    [
        html.H1("Company Map", style={"textAlign": "center"}),
        dcc.Graph(figure=fig, style={"height": "90vh"}),
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
