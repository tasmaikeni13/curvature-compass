"""Charged curvature oracles and a shared projected quadratic update."""
from dataclasses import dataclass
import time
import numpy as np


@dataclass
class Problem:
    H: np.ndarray
    g: np.ndarray
    family: str
    regime: str
    mu: float = 1.0


FAMILIES = ("flat", "diagonal", "rotated", "lowrank", "clustered", "toeplitz", "blocks", "needle")
REGIMES = ("isotropic", "error", "low", "high")
FIXED = ("coordinate", "haar", "rademacher", "cosine", "prior")
ADAPTIVE = ("krylov", "greedy_coordinate", "gradient_random", "gradient_cosine", "gradient_coordinate")


def problem(family, n, seed, regime="isotropic", condition=1000.0):
    rng = np.random.default_rng(seed)
    rotation = np.eye(n)
    if family in ("rotated", "lowrank", "clustered", "blocks"):
        rotation = np.linalg.qr(rng.normal(size=(n, n)))[0]
    if family == "flat":
        eigen = np.ones(n)
    elif family in ("diagonal", "rotated"):
        eigen = np.geomspace(1, condition, n)
        if family == "diagonal":
            # Independent orientation/order prevents index leakage.
            rotation = np.eye(n)[:, rng.permutation(n)]
    elif family == "lowrank":
        eigen = np.ones(n)
        rank = max(2, n // 8)
        eigen[-rank:] = np.geomspace(10, condition, rank)
    elif family == "clustered":
        eigen = np.resize(np.array([1.0, np.sqrt(condition), condition]), n)
    elif family == "toeplitz":
        H = 0.95 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        eigen, rotation = np.linalg.eigh(H)
        eigen /= eigen.min()
    elif family == "blocks":
        # Partially coordinate structured, unlike the fully rotated family.
        half = n // 2
        rotation = np.eye(n)
        rotation[:half, :half] = np.linalg.qr(rng.normal(size=(half, half)))[0]
        eigen = np.r_[np.geomspace(1, condition, half), np.ones(n-half)]
    elif family == "needle":
        eigen = np.ones(n)
        eigen[-1] = condition
        rotation = np.eye(n)[:, rng.permutation(n)]
    else:
        raise ValueError(family)
    H = (rotation * eigen) @ rotation.T
    if regime == "isotropic":
        g = rng.normal(size=n)
    elif regime == "error":
        g = H @ rng.normal(size=n)
    else:
        indices = np.argsort(eigen)
        chosen = indices[:max(1, n//8)] if regime == "low" else indices[-max(1, n//8):]
        g = rotation[:, chosen] @ rng.normal(size=len(chosen))
    g /= np.linalg.norm(g)
    return Problem(H=H, g=g, family=family, regime=regime)


class VectorOracle:
    """The sole interface an acquisition policy has to curvature."""
    def __init__(self, matvec, n, noise=0.0, seed=0):
        self._matvec = matvec
        self.n = n
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.calls = 0
        self.seconds = 0.0

    def __call__(self, v):
        start = time.perf_counter()
        y = self._matvec(v)
        if self.noise:
            y = y + self.noise * self.rng.normal(size=self.n) / np.sqrt(self.n)
        self.calls += 1
        self.seconds += time.perf_counter() - start
        return y


def fixed_stream(n, kind, seed, prior=None, size=None):
    rng = np.random.default_rng(seed)
    size = n if size is None else size
    if kind == "coordinate":
        return np.eye(n, size)
    if kind == "haar":
        return np.linalg.qr(rng.normal(size=(size, n)).T)[0]
    if kind == "rademacher":
        return np.linalg.qr(rng.choice([-1.0, 1.0], size=(size, n)).T)[0]
    if kind == "cosine":
        columns = np.arange(size)
        stream = np.cos(np.pi * (np.arange(n)[:, None]+0.5) * columns / n)*np.sqrt(2/n)
        stream[:, 0] /= np.sqrt(2)
        return stream
    if kind == "prior":
        if prior is None:
            raise ValueError("A prior stream must be learned on independent calibration data")
        return prior[:, :size]
    raise ValueError(kind)


def orthogonalize(v, Q):
    v = np.array(v, dtype=float, copy=True)
    # Reorthogonalization is charged as overhead, not an HVP.
    for _ in range(2):
        v -= Q @ (Q.T @ v)
    norm = np.linalg.norm(v)
    return None if norm < 1e-11 else v / norm


def acquire(oracle, g, alignment, cap, seed=0, stopping="fixed", tolerance=0.1,
            cost=0.01, mu=1.0, min_count=1, prior=None, safety_floor=1e-9):
    """Return a nested path and actual step using only g and charged HVPs.

    All adaptive count variants can be read from the SAME full exact path for
    simulation efficiency. Their returned prefix calls/seconds are used in cost
    accounting; the full-path reuse is disclosed by the runner. Direct stopped
    execution is checked separately.
    """
    start = time.perf_counter()
    n = len(g)
    cap = min(cap, n)
    Q, Y = np.empty((n, 0)), np.empty((n, 0))
    stream = fixed_stream(n, alignment, seed, prior, cap) if alignment in FIXED else None
    completion_kind = {"gradient_random":"haar", "gradient_cosine":"cosine", "gradient_coordinate":"coordinate"}.get(alignment)
    completion_stream = fixed_stream(n, completion_kind, seed, size=cap) if completion_kind else None
    residual = g.copy()
    previous_decrease = 0.0
    path = []
    used_coordinates = set()
    completion_index = 0
    for index in range(cap):
        if stream is not None:
            v = stream[:, index]
        elif alignment == "krylov":
            v = residual
        elif alignment == "greedy_coordinate":
            scores = np.abs(residual).copy()
            for used in used_coordinates:
                scores[used] = -1
            chosen = int(np.argmax(scores))
            used_coordinates.add(chosen)
            v = np.eye(1, n, chosen)[0]
        elif completion_stream is not None:
            if index == 0:
                v = g
            else:
                v = None
                while completion_index < completion_stream.shape[1]:
                    candidate = completion_stream[:, completion_index]
                    completion_index += 1
                    v = orthogonalize(candidate, Q)
                    if v is not None:
                        break
                if v is None:
                    break
        else:
            raise ValueError(alignment)
        v = orthogonalize(v, Q)
        if v is None:
            # Exact Krylov invariant-subspace exhaustion is not count adaptation.
            break
        y = oracle(v)
        Q = np.column_stack((Q, v))
        Y = np.column_stack((Y, y))
        G = Q.T @ Y
        G = (G + G.T) / 2
        values, vectors = np.linalg.eigh(G)
        clamped = np.maximum(values, safety_floor)
        rhs = Q.T @ g
        coefficients = -(vectors @ ((vectors.T @ rhs) / clamped))
        step = Q @ coefficients
        residual = g + Y @ coefficients
        decrease = float(-g @ step - 0.5 * coefficients @ G @ coefficients)
        elapsed = time.perf_counter() - start
        row = {"count": index+1, "decrease_model": decrease,
               "gain": decrease - previous_decrease,
               "residual_norm": float(np.linalg.norm(residual)),
               "certificate": float(residual @ residual / (2*mu)),
               "seconds": elapsed, "hvp_seconds": oracle.seconds,
               "clipped_eigenvalues": int(np.sum(values < safety_floor)),
               "step": step.copy()}
        path.append(row)
        previous_decrease = decrease
        if index+1 >= min_count and should_stop(row, stopping, tolerance, cost, g):
            break
    return path


def should_stop(row, stopping, tolerance, cost, g):
    if stopping == "fixed":
        return False
    if stopping == "residual":
        return row["residual_norm"] <= tolerance * np.linalg.norm(g)
    if stopping == "gain":
        return row["gain"] <= tolerance * max(row["decrease_model"], 1e-15)
    if stopping == "cost":
        # A declared gradient/lower-bound scale, not the hidden Newton optimum.
        return row["gain"] <= cost * float(g @ g) / 2
    raise ValueError(stopping)


def prefix(path, cap, stopping, tolerance, cost, g, min_count=1):
    if not path:
        raise ValueError("empty acquisition path")
    for row in path[:cap]:
        if row["count"] >= min_count and should_stop(row, stopping, tolerance, cost, g):
            return row
    return path[min(cap, len(path))-1]


def cg(oracle, g, cap, tolerance=0.0):
    """Independent conventional CG reference; solve Hs = -g."""
    s = np.zeros_like(g)
    r, p = -g.copy(), -g.copy()
    for _ in range(cap):
        Ap = oracle(p)
        denominator = p @ Ap
        if denominator <= 0:
            break
        alpha = (r @ r) / denominator
        s += alpha * p
        next_r = r - alpha * Ap
        if np.linalg.norm(next_r) <= max(tolerance*np.linalg.norm(g), 1e-13):
            break
        beta = (next_r @ next_r) / (r @ r)
        p = next_r + beta*p
        r = next_r
    return s


def nystrom_step(oracle, g, rank, seed, ridge=1.0):
    """Low-rank Nyström inverse with a DECLARED residual ridge.

    This is an adapted low-rank preconditioning baseline, not a reproduction
    of Frangella et al.'s tuned Nyström PCG algorithm.
    """
    Q = fixed_stream(len(g), "haar", seed)[:, :rank]
    Y = np.column_stack([oracle(q) for q in Q.T])
    W = (Q.T @ Y + Y.T @ Q) / 2
    values, vectors = np.linalg.eigh(W)
    factor = Y @ (vectors / np.sqrt(np.maximum(values, 1e-12)))
    U, singular, _ = np.linalg.svd(factor, full_matrices=False)
    eigen = singular**2
    return -g/ridge + U @ ((1/ridge - 1/(eigen+ridge)) * (U.T @ g))
