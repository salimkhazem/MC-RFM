"""2D toy experiment for Euclidean vs hyperbolic transport for theory verification"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.geometry.poincare import expmap0, logmap0, mobius_add, project_to_ball


def generate_toy(n: int = 800, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, size=n)
    source = rng.normal(loc=0.0, scale=0.25, size=(n, 2))
    centers = np.array([[1.8, 0.9], [1.8, -0.9], [-1.8, 0.9], [-1.8, -0.9]], dtype=np.float32)
    target = centers[labels] + rng.normal(loc=0.0, scale=0.12, size=(n, 2))
    return source.astype(np.float32), target.astype(np.float32), labels.astype(np.int64), centers


def euclidean_transport(source: np.ndarray, labels: np.ndarray, centers: np.ndarray, nfe: int = 15):
    z = torch.from_numpy(source).clone()
    centers_t = torch.from_numpy(centers)
    traj = [z.clone().numpy()]
    dt = 1.0 / nfe
    for _ in range(nfe):
        v = centers_t[torch.from_numpy(labels)] - z
        z = z + dt * v
        traj.append(z.clone().numpy())
    return z.numpy(), traj


def hyperbolic_transport(source: np.ndarray, labels: np.ndarray, centers: np.ndarray, c: float = 0.5, nfe: int = 15):
    src_t = torch.from_numpy(source) * 0.35
    ctr_t = torch.from_numpy(centers) * 0.35
    z = expmap0(src_t, c=c)
    tgt = expmap0(ctr_t, c=c)
    traj = [z.clone().numpy()]
    dt = 1.0 / nfe
    label_t = torch.from_numpy(labels)
    for _ in range(nfe):
        delta = mobius_add(-z, tgt[label_t], c=c)
        v = logmap0(delta, c=c)
        z = project_to_ball(z + dt * v, c=c, eps=1.0e-5)
        traj.append(z.clone().numpy())
    return z.numpy(), traj, tgt.numpy()


def branch_separation(points: np.ndarray, labels: np.ndarray) -> float:
    centers = []
    intra = []
    for cls in np.unique(labels):
        p = points[labels == cls]
        c = p.mean(axis=0)
        centers.append(c)
        intra.append(np.mean(np.linalg.norm(p - c[None, :], axis=1)))
    centers = np.stack(centers, axis=0)
    inter = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            inter.append(np.linalg.norm(centers[i] - centers[j]))
    return float(np.mean(inter) / max(np.mean(intra), 1.0e-9))


def plot_quiver_euclidean(ax, centers: np.ndarray):
    gx, gy = np.meshgrid(np.linspace(-2.5, 2.5, 25), np.linspace(-2.0, 2.0, 25))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    d = ((grid[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    nearest = d.argmin(axis=1)
    v = centers[nearest] - grid
    ax.quiver(grid[:, 0], grid[:, 1], v[:, 0], v[:, 1], angles="xy", scale_units="xy", scale=15, alpha=0.8)
    ax.set_title("Euclidean Field")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.0, 2.0)


def plot_quiver_hyperbolic(ax, tgt_ball: np.ndarray, c: float = 0.5):
    gx, gy = np.meshgrid(np.linspace(-1.2, 1.2, 25), np.linspace(-1.2, 1.2, 25))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)
    grid_t = torch.from_numpy(grid)
    grid_t = project_to_ball(grid_t, c=c, eps=1.0e-4)
    tgt_t = torch.from_numpy(tgt_ball)
    d = ((grid_t.unsqueeze(1) - tgt_t.unsqueeze(0)) ** 2).sum(dim=-1)
    nearest = d.argmin(dim=1)
    delta = mobius_add(-grid_t, tgt_t[nearest], c=c)
    v = logmap0(delta, c=c).numpy()
    g = grid_t.numpy()
    ax.quiver(g[:, 0], g[:, 1], v[:, 0], v[:, 1], angles="xy", scale_units="xy", scale=10, alpha=0.8)
    circle = plt.Circle((0, 0), radius=1.0 / (c**0.5), fill=False, linestyle="--", linewidth=1.0)
    ax.add_patch(circle)
    ax.set_title("Hyperbolic Field (Poincaré)")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)


def main() -> None:
    out_dir = Path("results/toy")
    out_dir.mkdir(parents=True, exist_ok=True)

    source, target, labels, centers = generate_toy()
    eu_final, eu_traj = euclidean_transport(source, labels, centers, nfe=15)
    hy_final, hy_traj, tgt_ball = hyperbolic_transport(source, labels, centers, c=0.5, nfe=15)
    sep_e = branch_separation(eu_final, labels)
    sep_h = branch_separation(hy_final, labels)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    plot_quiver_euclidean(axes[0], centers)
    plot_quiver_hyperbolic(axes[1], tgt_ball, c=0.5)
    axes[2].scatter(eu_final[:, 0], eu_final[:, 1], c=labels, s=8, cmap="tab10", alpha=0.7, label="Euclidean")
    axes[2].scatter(hy_final[:, 0], hy_final[:, 1], c=labels, s=8, marker="x", cmap="tab10", alpha=0.7, label="Hyperbolic")
    axes[2].set_title(f"Final Samples\nSep(E)={sep_e:.2f}, Sep(H)={sep_h:.2f}")
    axes[2].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    fig.savefig(out_dir / "toy_transport.png", dpi=220)
    fig.savefig(out_dir / "toy_transport.pdf")
    plt.close(fig)

    # Trajectory figure.
    sel = np.linspace(0, len(source) - 1, 40, dtype=int)
    fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
    for idx in sel:
        eu_points = np.stack([t[idx] for t in eu_traj], axis=0)
        ax2.plot(eu_points[:, 0], eu_points[:, 1], color="tab:blue", alpha=0.15)
    ax2.scatter(eu_final[:, 0], eu_final[:, 1], c=labels, s=6, cmap="tab10", alpha=0.5)
    ax2.set_title("Euclidean Transport Trajectories")
    ax2.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    fig2.savefig(out_dir / "toy_trajectories_euclidean.png", dpi=220)
    fig2.savefig(out_dir / "toy_trajectories_euclidean.pdf")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(1, 1, figsize=(5, 5))
    for idx in sel:
        hy_points = np.stack([t[idx] for t in hy_traj], axis=0)
        ax3.plot(hy_points[:, 0], hy_points[:, 1], color="tab:orange", alpha=0.15)
    ax3.scatter(hy_final[:, 0], hy_final[:, 1], c=labels, s=6, cmap="tab10", alpha=0.5)
    circle = plt.Circle((0, 0), radius=1.0 / (0.5**0.5), fill=False, linestyle="--", linewidth=1.0)
    ax3.add_patch(circle)
    ax3.set_title("Hyperbolic Transport Trajectories")
    ax3.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    fig3.savefig(out_dir / "toy_trajectories_hyperbolic.png", dpi=220)
    fig3.savefig(out_dir / "toy_trajectories_hyperbolic.pdf")
    plt.close(fig3)

    metrics_path = out_dir / "toy_metrics.txt"
    metrics_path.write_text(
        f"branch_separation_euclidean={sep_e:.6f}\nbranch_separation_hyperbolic={sep_h:.6f}\n",
        encoding="utf-8",
    )
    print(f"Wrote toy outputs to: {out_dir}")


if __name__ == "__main__":
    main()

