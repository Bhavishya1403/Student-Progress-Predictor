"""Compatibility import for the canonical leakage-safe trainer."""

from src.model_training import train_model


def run_training(**kwargs):
    return train_model(**kwargs)
