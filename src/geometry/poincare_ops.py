"""Compatibility module re-exporting Poincaré operations."""

from src.geometry.poincare import (
    ball_radius,
    dist,
    exp_map_0,
    geodesic,
    hyperbolic_norm,
    log_map,
    mobius_add,
    mobius_scalar_mul,
    project_to_ball,
)

__all__ = [
    "ball_radius",
    "dist",
    "exp_map_0",
    "geodesic",
    "hyperbolic_norm",
    "log_map",
    "mobius_add",
    "mobius_scalar_mul",
    "project_to_ball",
]

