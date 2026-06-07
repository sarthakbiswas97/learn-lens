"""UMAP scatter plot."""

from __future__ import annotations

import gradio as gr

from learnlens.config import LearnLensConfig
from learnlens.models.connector import ConnectorModel
from learnlens.storage.database import Database


def create(db: Database, connector: ConnectorModel, config: LearnLensConfig) -> None:
    """Build the Knowledge Map tab."""
    with gr.Tab("Knowledge Map"):
        gr.Markdown("## Knowledge Map")
        gr.Markdown("2D visualization of your content embeddings.")

        plot = gr.Plot(label="UMAP Projection")
        refresh_btn = gr.Button("Refresh")
        refresh_btn.click(fn=lambda: _load_plot(db, connector, config), outputs=plot)


def _load_plot(
    db: Database, connector: ConnectorModel, config: LearnLensConfig
) -> dict | None:
    try:
        import numpy as np
        import plotly.express as px
        import umap

        from learnlens.storage.queries import get_all_embeddings

        content_ids, matrix = get_all_embeddings(db.connection)
        if matrix.size == 0 or len(content_ids) < 5:
            return None

        reducer = umap.UMAP(n_neighbors=min(15, len(content_ids) - 1), random_state=42)
        embedding_2d = reducer.fit_transform(matrix)

        fig = px.scatter(
            x=embedding_2d[:, 0],
            y=embedding_2d[:, 1],
            title="Knowledge Map",
            labels={"x": "UMAP 1", "y": "UMAP 2"},
        )
        return fig
    except Exception:
        return None
