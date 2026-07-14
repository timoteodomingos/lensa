from pathlib import Path

import datamapplot
import duckdb
import numpy as np
import pandas as pd
from palette import make_position_palette

con = duckdb.connect("db/lensa.db")

# left join on financials later?
df = con.sql("""
    SELECT
        ct.id,
        ct.label_layer_0,
        ct.label_layer_1,
        ct.label_layer_2,
        ct.label_layer_3,
        ct.label_layer_4,
        cs.short_summary,
        cxy.x,
        cxy.y,
        br.city as municipality,
        br.name,
        cw.url as website,
        cf.revenue,
        cf.net_profit

    FROM company_topics ct
    JOIN company_summaries cs ON ct.id = cs.id
    JOIN company_xy_coords cxy ON ct.id = cxy.id
    JOIN brreg_data br ON ct.id = br.id
    JOIN websites cw ON ct.id = cw.id
    LEFT JOIN (
        SELECT DISTINCT ON (id)
            id, revenue, net_profit, year
        FROM company_financials
        WHERE year IN ('2024', '2025')
        ORDER BY id, year DESC
    ) cf ON ct.id = cf.id
""").df()

print(f"Loaded {len(df)} companies")

coords = np.ascontiguousarray(df[["x", "y"]].values, dtype=np.float32)

label_columns = sorted(c for c in df.columns if c.startswith("label_layer_"))
topic_name_vectors = [df[c].values for c in label_columns]

hover_template = Path("app/templates/hover_card.html").read_text()
custom_css = Path("app/templates/hover_card.css").read_text()
tooltip_css = Path("app/templates/tooltip.css").read_text()

revenue_log = np.log1p(df["revenue"].fillna(0).clip(lower=0).values)


def fmt_nok(val):
    if pd.isna(val):
        return "N/A"
    return f"NOK {int(val):,}".replace(",", "\u00a0")


def fmt_profit(val):
    if pd.isna(val):
        return "N/A"
    prefix = "▲" if val > 0 else "▼"
    return f"{prefix} NOK {abs(int(val)):,}".replace(",", "\u00a0")


extra_data = pd.DataFrame(
    {
        "company_name": df["name"].fillna("").values,
        "municipality": df["municipality"].fillna("").values,
        "short_summary": df["short_summary"].fillna("").values,
        "website": df["website"].fillna("").values,
        "revenue": df["revenue"].apply(fmt_nok).values,
        "net_profit": df["net_profit"].apply(fmt_profit).values,
    }
)

marker_color_array = make_position_palette(
    labels=df["label_layer_2"].values,
    coords=coords,
)


plot = datamapplot.create_interactive_plot(
    coords,
    *topic_name_vectors,
    title="Companies in Oslo (AS)",
    sub_title="Similar companies cluster together",
    hover_text=df["short_summary"].fillna("").tolist(),
    hover_text_html_template=hover_template,
    extra_point_data=extra_data,
    enable_search=True,
    tooltip_css=tooltip_css,
    custom_css=custom_css,
    marker_color_array=marker_color_array,
    colormap_rawdata=[revenue_log],
    colormap_metadata=[
        {
            "field": "revenue",
            "description": "Revenue (log scale)",
            "kind": "continuous",
            "cmap": "Greens",
        }
    ]
)

plot.save("norwegian_companies_map.html")
print("Saved norwegian_companies_map.html")

con.close()
