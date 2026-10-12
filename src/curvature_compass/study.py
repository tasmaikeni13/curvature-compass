"""Predeclared balanced Monte Carlo candidate search and held-out evaluation."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import concurrent.futures
import csv
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import scipy
from .core import (ADAPTIVE, FIXED, FAMILIES, REGIMES, VectorOracle,
                   acquire, prefix, problem)

ROOT = Path("analysis/results")
DIMS = (16, 48, 96)
CAPS = (2, 8, 16)
STOPS = (("fixed",0.0), ("residual",0.1), ("residual",0.3),
         ("residual",0.7), ("gain",0.01), ("gain",0.1),
         ("cost",0.01), ("cost",0.05))
FIELDS = ("stage", "seed", "family", "dimension", "regime", "alignment", "stopping",
          "tolerance", "cap", "count", "progress", "progress_per_hvp", "energy_error",
          "residual_norm", "certificate", "seconds", "hvp_seconds", "clipped_eigenvalues")


def seed_for(seed, family, n):
    return seed*1000000 + FAMILIES.index(family)*10000+n*10


def learn_prior():
    priors = {}
    for n in DIMS:
        covariance = np.zeros((n,n))
        for seed in range(1000,1024):
            for family in FAMILIES:
                for regime in REGIMES:
                    p = problem(family,n,seed_for(seed,family,n),regime)
                    covariance += np.outer(p.g,p.g)
        priors[str(n)] = np.linalg.eigh(covariance)[1][:,::-1]
    np.savez_compressed(ROOT / "offline_priors.npz", **priors)
    return priors


def task(args):
    stage, seed, priors = args
    output = []
    for n in DIMS:
        for family in FAMILIES:
            for regime in REGIMES:
                p = problem(family,n,seed_for(seed,family,n),regime)
                sstar = np.linalg.solve(p.H,-p.g)
                Fstar = float(-0.5*p.g@sstar)
                for alignment in FIXED+ADAPTIVE:
                    oracle = VectorOracle(lambda v:p.H@v,n)
                    path = acquire(oracle,p.g,alignment,max(CAPS),seed_for(seed,family,n)+17,
                                   prior=priors[str(n)])
                    for cap in CAPS:
                        for stopping,tolerance in STOPS:
                            cost = tolerance if stopping=="cost" else 0.01
                            row = prefix(path,cap,stopping,tolerance,cost,p.g)
                            s = row["step"]
                            actual = float(-p.g@s-0.5*s@p.H@s)
                            progress = actual/Fstar
                            if progress < -1e-7 or progress > 1+1e-7:
                                raise AssertionError((family,alignment,progress))
                            output.append({"stage":stage,"seed":seed,"family":family,
                                "dimension":n,"regime":regime,"alignment":alignment,
                                "stopping":stopping,"tolerance":tolerance,"cap":cap,
                                "count":row["count"],"progress":progress,
                                "progress_per_hvp":progress/row["count"],
                                "energy_error":float((s-sstar)@p.H@(s-sstar)/(2*Fstar)),
                                **{k:row[k] for k in FIELDS if k in row}})
    return output


def run(stage, workers=8, repeats=None):
    ROOT.mkdir(parents=True,exist_ok=True)
    if stage=="calibration":
        seeds=list(range(1000,1000+(repeats or 24)))
        priors=learn_prior()
    else:
        if not (ROOT/"selection.json").exists():
            raise ValueError("Calibrate and freeze selection before evaluation")
        seeds=list(range(2000,2000+(repeats or 64)))
        priors=dict(np.load(ROOT/"offline_priors.npz"))
    started=time.perf_counter()
    path=ROOT/(stage+".csv.gz")
    count=0
    with gzip.open(path,"wt",newline="",compresslevel=6) as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS)
        writer.writeheader()
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in pool.map(task,[(stage,seed,priors) for seed in seeds]):
                writer.writerows(batch)
                count+=len(batch)
                print(f"{stage}: {count} outcomes",flush=True)
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in Path("src/curvature_compass").glob("*.py")}
    manifest={"stage":stage,"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
      "seconds":time.perf_counter()-started,"seeds":seeds,"rows":count,
      "problem_instances":len(seeds)*len(DIMS)*len(FAMILIES)*len(REGIMES),
      "dimensions":DIMS,"caps":CAPS,"families":FAMILIES,"regimes":REGIMES,
      "stopping_candidates":STOPS,"workers":workers,"python":platform.python_version(),
      "numpy":np.__version__,"scipy":scipy.__version__,"platform":platform.platform(),
      "blas_threads":os.environ["OPENBLAS_NUM_THREADS"],"source_sha256":source_hashes,
      "raw_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
      "accounting":"Full acquisition paths reused across candidate caps/stopping rules; each policy charged its exact prefix count and measured prefix time, including full-cap stream setup. Candidate-search total cost is the actual full-path run cost; policy-prefix counts are not added to pretend these were separate oracle executions."}
    (ROOT/(stage+"_manifest.json")).write_text(json.dumps(manifest,indent=2)+"\n")
    if stage=="calibration":
        select()


def rows(stage):
    with gzip.open(ROOT/(stage+".csv.gz"),"rt") as stream:
        for r in csv.DictReader(stream):
            for key in FIELDS:
                if key not in ("stage","family","regime","alignment","stopping"):
                    r[key]=float(r[key])
            yield r


def select():
    groups={}
    for r in rows("calibration"):
        key=(r["alignment"],r["stopping"],r["tolerance"])
        groups.setdefault(key,[]).append(r["progress_per_hvp"])
    scores={key:float(np.mean(values)) for key,values in groups.items()}
    fixed=max((key for key in scores if key[0] in FIXED and key[1]=="fixed"),key=scores.get)[0]
    adaptive=max((key for key in scores if key[0] in ADAPTIVE and key[1]=="fixed"),key=scores.get)[0]
    # The count policy is common to C and D. This prevents confounding the cells
    # by giving each adaptive cell a different stopping objective.
    stopkeys=[(kind,tol) for kind,tol in STOPS if kind!="fixed"]
    best_stop=max(stopkeys,key=lambda k:(scores[(fixed,*k)]+scores[(adaptive,*k)])/2)
    selection={"criterion":"balanced mean Newton-progress per charged HVP across all calibration populations and caps",
      "calibration_seeds":list(range(1000,1024)),"main_cap":8,"fixed_alignment":fixed,
      "adaptive_alignment":adaptive,"stopping":best_stop[0],"tolerance":best_stop[1],
      "configurations":{"A":{"alignment":fixed,"stopping":"fixed","tolerance":0.0},
                        "B":{"alignment":adaptive,"stopping":"fixed","tolerance":0.0},
                        "C":{"alignment":fixed,"stopping":best_stop[0],"tolerance":best_stop[1]},
                        "D":{"alignment":adaptive,"stopping":best_stop[0],"tolerance":best_stop[1]}},
      "candidate_scores":[{"alignment":k[0],"stopping":k[1],"tolerance":k[2],"score":v} for k,v in scores.items()]}
    (ROOT/"selection.json").write_text(json.dumps(selection,indent=2)+"\n")
    print(json.dumps({k:v for k,v in selection.items() if k!="candidate_scores"},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["calibration","evaluation"])
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--repeats",type=int)
    args=parser.parse_args()
    run(args.stage,args.workers,args.repeats)
