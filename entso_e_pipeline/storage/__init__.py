"""Persistent data stores used by ingestion and forecasting."""

from .supabase import SupabaseRawStore

__all__ = ["SupabaseRawStore"]
