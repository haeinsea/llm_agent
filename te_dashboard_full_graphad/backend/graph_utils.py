from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd

from .config import TE_VAR_PROCESS_MAP


def load_process_graph() -> Tuple[nx.Graph, Dict[str, str]]:
    """Load the variable-to-process mapping and build a bipartite process graph.

    Expected CSV columns: ``var`` and ``process``.
    """
    g = nx.DiGraph()
    var_to_proc: Dict[str, str] = {}

    path = Path(TE_VAR_PROCESS_MAP)
    if not path.exists():
        return g, var_to_proc

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if not {"var", "process"}.issubset(df.columns):
        return g, var_to_proc

    for _, row in df.iterrows():
        v = str(row["var"]).strip()
        p = str(row["process"]).strip()
        var_to_proc[v] = p
        if not g.has_node(p):
            g.add_node(p, type="process")
        if not g.has_node(v):
            g.add_node(v, type="variable")
        g.add_edge(p, v)

    return g, var_to_proc


def build_sample_subgraph(g: nx.Graph, top_vars: List[str], hops: int = 1) -> nx.Graph:
    if g is None or g.number_of_nodes() == 0:
        return nx.Graph()

    nodes = set()
    for v in top_vars:
        if v not in g:
            continue
        nodes.add(v)
        frontier = {v}
        for _ in range(hops):
            neigh = set()
            for u in frontier:
                neigh.update(g.neighbors(u))
            frontier = neigh
            nodes.update(frontier)

    return g.subgraph(nodes).copy()


def graph_to_json(g: nx.Graph):
    """Convert a process/variable graph into Cytoscape-friendly JSON."""
    # --- Load variable display labels ---
    var_ko_map: Dict[str, str] = {}
    path = Path(TE_VAR_PROCESS_MAP)
    if path.exists():
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip() for c in df.columns]
            if {"var", "ko"}.issubset(df.columns):
                for _, row in df.iterrows():
                    v = str(row["var"]).strip()
                    ko = str(row["ko"]).strip()
                    # Keep only the short label before parentheses.
                    if "(" in ko:
                        ko = ko.split("(", 1)[0].strip()
                    var_ko_map[v] = ko if ko else v
        except Exception:
            # Fall back to raw variable ids if the mapping file cannot be read.
            var_ko_map = {}

    nodes = []
    edges = []
    for n, data in g.nodes(data=True):
        node_type = data.get("type", "variable")
        label = n
        if node_type == "variable":
            label = var_ko_map.get(n, n)
        nodes.append(
            {
                "id": n,
                "label": label,
                "type": node_type,
            }
        )

    for u, v, data in g.edges(data=True):
        edges.append(
            {
                "source": u,
                "target": v,
                "weight": float(data.get("weight", 1.0)),
            }
        )
    return {"nodes": nodes, "edges": edges}
