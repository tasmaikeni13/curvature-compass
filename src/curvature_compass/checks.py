"""Scientific invariants and deliberate counterexamples, not mock tests."""
import json
from pathlib import Path
import numpy as np
from .core import VectorOracle, acquire, cg, fixed_stream, prefix, problem, FAMILIES


def check():
    errors = {"cg_energy_relative": [], "residual_certificate_violation": [],
              "nested_decrease_violation": [], "schur_gain_relative": [],
              "galerkin_orthogonality": [], "stopped_prefix_difference": []}
    cases = 0
    for seed in range(20):
        for family in FAMILIES:
            p = problem(family, 16, seed)
            optimal = np.linalg.solve(p.H, -p.g)
            Fstar = float(-0.5*p.g @ optimal)
            for alignment in ("haar", "coordinate", "krylov", "greedy_coordinate"):
                path = acquire(VectorOracle(lambda v:p.H@v, 16), p.g, alignment, 8, seed)
                Q = []
                previous = None
                for row in path:
                    s = row["step"]
                    r = p.g+p.H@s
                    gap = 0.5*(s-optimal)@p.H@(s-optimal)
                    errors["residual_certificate_violation"].append(max(0.0, gap-row["certificate"]))
                    if previous is not None:
                        errors["nested_decrease_violation"].append(max(0.0, previous["decrease_model"]-row["decrease_model"]))
                    previous = row
                if alignment == "krylov":
                    conventional = cg(VectorOracle(lambda v:p.H@v, 16), p.g, len(path))
                    difference = conventional-path[-1]["step"]
                    errors["cg_energy_relative"].append(float(difference@p.H@difference/max(Fstar,1e-15)))
                chosen = prefix(path, 8, "residual", 0.3, 0.01, p.g)
                direct = acquire(VectorOracle(lambda v:p.H@v, 16), p.g, alignment, 8, seed, "residual", 0.3)
                errors["stopped_prefix_difference"].append(float(np.linalg.norm(chosen["step"]-direct[-1]["step"])))
                cases += 1
            Q = fixed_stream(16, "haar", seed, size=8)
            for k in range(1,8):
                U, v = Q[:,:k], Q[:,k]
                G = U.T@p.H@U
                s = -U@np.linalg.solve(G,U.T@p.g)
                r = p.g+p.H@s
                w = U.T@p.H@v
                schur = v@p.H@v-w@np.linalg.solve(G,w)
                predicted = (v@r)**2/(2*schur)
                V = Q[:,:k+1]
                snew = -V@np.linalg.solve(V.T@p.H@V,V.T@p.g)
                observed = (-p.g@snew-0.5*snew@p.H@snew)-(-p.g@s-0.5*s@p.H@s)
                errors["schur_gain_relative"].append(abs(predicted-observed)/max(Fstar,1e-15))
                errors["galerkin_orthogonality"].append(float(np.linalg.norm(U.T@r)))
    # The same scalar observations hide off-diagonal entries, unlike HVPs.
    H0, H1 = np.eye(2), np.array([[1,.9],[.9,1]])
    assert np.allclose(np.diag(H0),np.diag(H1))
    assert not np.allclose(H0[:,0],H1[:,0])
    g = np.array([0.,1.])
    first = 0.0
    second = 0.5
    both = 0.5*g@np.linalg.solve(H1,g)
    assert both-first > second
    # Marginal stopping misses a valuable late fixed direction on an SPD identity.
    g = np.r_[np.zeros(15),1.]
    early = acquire(VectorOracle(lambda v:v.copy(),16), g,"coordinate",16,stopping="gain")
    safe = acquire(VectorOracle(lambda v:v.copy(),16), g,"coordinate",16,stopping="residual")
    assert len(early)==1 and early[-1]["decrease_model"]==0
    assert len(safe)==16 and np.isclose(safe[-1]["decrease_model"],.5)
    # Scalar polarization reconstructs a dense symmetric matrix exactly.
    rng = np.random.default_rng(777)
    H = rng.normal(size=(12,12)); H=(H+H.T)/2
    reconstructed = np.diag(np.diag(H))
    for i in range(12):
        for j in range(i+1,12):
            v=np.eye(12)[i]+np.eye(12)[j]
            reconstructed[i,j]=reconstructed[j,i]=(v@H@v-H[i,i]-H[j,j])/2
    assert np.allclose(H,reconstructed)
    maxima = {key:float(max(values,default=0.0)) for key,values in errors.items()}
    # Conventional three-term CG and reorthogonalized Galerkin have different
    # roundoff on ill-conditioned clustered spectra. Compare energy at 1e-7,
    # rather than falsely demanding identical floating-point algorithms.
    assert maxima["cg_energy_relative"] < 1e-7, maxima
    assert maxima["residual_certificate_violation"] < 1e-9, maxima
    assert maxima["nested_decrease_violation"] < 1e-9, maxima
    assert maxima["schur_gain_relative"] < 1e-10, maxima
    assert maxima["galerkin_orthogonality"] < 1e-10, maxima
    assert maxima["stopped_prefix_difference"] == 0.0, maxima
    report = {"status":"passed", "acquisition_cases":cases, "maxima":maxima,
              "cg_energy_tolerance":1e-7,
              "counterexamples":{"non_submodular_gain":float(both-second),
              "blind_prefix_gain_stop_count":len(early),"certificate_stop_count":len(safe)},
              "scalar_polarization_max_error":float(np.max(np.abs(H-reconstructed)))}
    Path("analysis/results").mkdir(exist_ok=True)
    Path("analysis/results/invariants.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    check()
