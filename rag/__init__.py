"""
Package init for the rag module.
Exposes the public-facing build_graph function as the top-level API.
"""

from rag.graph import build_graph

__all__ = ["build_graph"]
