"""Seed-clustered analysis of frozen Monte Carlo selections."""
import collections
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .study import rows, ROOT


def interval(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(424242)
    resamples=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(axis=1)
    return {"mean":float(values.mean()),"ci95":[float(x) for x in np.quantile(resamples,[.025,.975])],"clusters":len(values)}


def summarize(records,metric):
    by_seed=collections.defaultdict(list)
    for r in records:
        by_seed[int(r["seed"])].append(r[metric])
    return interval([np.mean(values) for _,values in sorted(by_seed.items())])


def make_report():
    selection=json.loads((ROOT/"selection.json").read_text())
    configs=selection["configurations"]
    main=collections.defaultdict(list)
    fixed_curves=collections.defaultdict(list)
    candidates=collections.defaultdict(list)
    for r in rows("evaluation"):
        key=(r["alignment"],r["stopping"],r["tolerance"],int(r["cap"]))
        candidates[key].append([r["progress"],r["count"],r["progress_per_hvp"]])
        if r["stopping"]=="fixed" and r["alignment"] in (selection["fixed_alignment"],selection["adaptive_alignment"]):
            fixed_curves[(r["alignment"],int(r["cap"]))].append(r)
        if r["cap"]!=selection["main_cap"]:
            continue
        for label,config in configs.items():
            if all(r[k]==v for k,v in config.items()):
                main[label].append(r)
    metrics=("progress","count","progress_per_hvp","energy_error","seconds","hvp_seconds")
    result={"population":"eight fixed families x three dimensions x four gradient regimes, balanced",
            "inference":"percentile bootstrap of 64 independent root-seed clusters; 2000 resamples; descriptive 95% intervals",
            "selection":{k:v for k,v in selection.items() if k!="candidate_scores"},
            "cells":{label:{metric:summarize(records,metric) for metric in metrics} for label,records in main.items()},
            "by_family":{},"by_dimension":{},"by_regime":{},"paired_effects":{},"cost_sensitivity":[]}
    for label,records in main.items():
        result["cells"][label]["ratio_of_means_progress_per_hvp"]=float(np.mean([r["progress"] for r in records])/np.mean([r["count"] for r in records]))
        result["cells"][label]["invariant_exhaustion_fraction"]=float(np.mean([r["count"]<r["cap"] for r in records])) if label in ("A","B") else None
    for axis,target in (("family","by_family"),("dimension","by_dimension"),("regime","by_regime")):
        for value in sorted({r[axis] for r in main["A"]}):
            result[target][str(value)]={label:{metric:summarize([r for r in records if r[axis]==value],metric)
                for metric in ("progress","count","progress_per_hvp")} for label,records in main.items()}
    paired=collections.defaultdict(dict)
    for label,records in main.items():
        for r in records:
            paired[(r["seed"],r["family"],r["dimension"],r["regime"])][label]=r
    for metric in ("progress","count","progress_per_hvp"):
        for name,weights in {"alignment_B_minus_A":{"B":1,"A":-1},
                "count_C_minus_A":{"C":1,"A":-1},"joint_D_minus_A":{"D":1,"A":-1},
                "interaction_D_minus_B_minus_C_plus_A":{"D":1,"B":-1,"C":-1,"A":1}}.items():
            effects=[{"seed":key[0],"value":sum(weight*vals[label][metric] for label,weight in weights.items())}
                     for key,vals in paired.items()]
            result["paired_effects"][metric+":"+name]=summarize(effects,"value")
    # At each adaptive mean count, compare the linear interpolation of independently
    # randomized fixed-budget policies. This is a randomized mixture of caps, not
    # an illicit fractional HVP, and only exists inside the measured support.
    result["matched_mean_count"]={}
    for adaptive_label,fixed_label in (("C","A"),("D","B")):
        alignment=configs[fixed_label]["alignment"]
        points=sorted([(np.mean([r["count"] for r in records]),np.mean([r["progress"] for r in records]))
                      for (a,cap),records in fixed_curves.items() if a==alignment])
        adaptive_count=result["cells"][adaptive_label]["count"]["mean"]
        entry={"adaptive_count":adaptive_count,"fixed_frontier":points,
               "interpretation":"interpolation is a pre-randomized mixture of fixed caps, descriptive only; no paired interval for this derived frontier"}
        if points[0][0]<=adaptive_count<=points[-1][0]:
            entry["fixed_mixture_progress"]=float(np.interp(adaptive_count,[p[0] for p in points],[p[1] for p in points]))
            entry["adaptive_minus_mixture"]=result["cells"][adaptive_label]["progress"]["mean"]-entry["fixed_mixture_progress"]
        else:
            entry["status"]="outside observed fixed-count support; no extrapolated comparison"
        result["matched_mean_count"][adaptive_label]=entry
    for ratio in (0.,.1,.5,1.,2.,5.,10.):
        scores={label:float(np.mean([r["progress"]/(1+ratio*r["count"]) for r in records])) for label,records in main.items()}
        result["cost_sensitivity"].append({"hvp_cost_in_gradient_units":ratio,"mean_progress_per_modeled_work":scores,"best_cell":max(scores,key=scores.get)})
    summaries=[]
    for key,records in candidates.items():
        values=np.mean(records,axis=0)
        summaries.append({"alignment":key[0],"stopping":key[1],"tolerance":key[2],"cap":key[3],
            "progress":float(values[0]),"count":float(values[1]),"mean_progress_per_hvp":float(values[2])})
    (ROOT/"candidate_summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    (ROOT/"main_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    figure,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    labels=list("ABCD")
    means=[result["cells"][label]["progress"]["mean"] for label in labels]
    cis=[result["cells"][label]["progress"]["ci95"] for label in labels]
    axes[0].bar(labels,means,color=["#49658c","#bd6b42","#5c9685","#7d5a8f"])
    axes[0].errorbar(labels,means,yerr=np.array([[m-c[0] for m,c in zip(means,cis)],[c[1]-m for m,c in zip(means,cis)]]),fmt="none",color="black",capsize=3)
    axes[0].set(ylabel="Fraction of exact Newton decrease",title="Frozen quartet, cap 8",ylim=(0,1.05))
    axes[1].bar(labels,[result["cells"][label]["count"]["mean"] for label in labels],color="#5c9685")
    axes[1].set(ylabel="Mean charged HVP count",title="Effective count, including rank exhaustion")
    for alignment,style in ((selection["fixed_alignment"],"o-"),(selection["adaptive_alignment"],"s-")):
        points=sorted([(np.mean([r["count"] for r in records]),np.mean([r["progress"] for r in records])) for (a,cap),records in fixed_curves.items() if a==alignment])
        axes[2].plot([p[0] for p in points],[p[1] for p in points],style,label=alignment)
    for label in ("C","D"):
        axes[2].scatter(result["cells"][label]["count"]["mean"],result["cells"][label]["progress"]["mean"],s=60,label=label)
    axes[2].set(xlabel="Mean charged HVPs",ylabel="Newton decrease fraction",title="Measured cost/progress frontier")
    axes[2].legend(fontsize=8)
    figure.savefig(ROOT/"main_comparison.png",dpi=180)
    plt.close(figure)
    lines=["# Held-out Phase 1 Monte Carlo results","",
      "The quartet was frozen in commit `6b44ca9` using 24 calibration seeds. The",
      "held-out set has 64 root seeds and 6,144 problem instances. Candidate policies",
      "share full paths for efficient evaluation; counts and times use each exact prefix.","",
      "Confidence intervals resample root seeds, preserving within-seed dependence.",
      "They describe the declared balanced synthetic population, not neural networks.","",
      "| Cell | Mean Newton decrease fraction (95% CI) | Mean HVPs | Mean fraction/HVP |",
      "|---|---|---|---|"]
    for label in labels:
        c=result["cells"][label]; p=c["progress"]
        lines.append(f"| {label} | {p['mean']:.4f} [{p['ci95'][0]:.4f}, {p['ci95'][1]:.4f}] | {c['count']['mean']:.3f} | {c['progress_per_hvp']['mean']:.4f} |")
    lines.extend(["","![Comparison](results/main_comparison.png)","",
      "Fixed alignment uses the offline PCA prior; adaptive alignment uses gradient",
      "and residual Krylov expansion. C/D stop at relative residual 0.7 or cap eight.",
      "This loose tolerance was selected for progress per HVP; it is not a guarantee",
      "of a small relative Newton error. More stringent tolerances remain in the raw",
      "candidate results and cost/progress analysis.","",
      "A fixed requested count can use fewer HVPs after exact Krylov-space exhaustion.",
      "This is a numerical rank limit, not evidence of an adaptive count policy in B.","",
      "| Paired progress effect | Mean | Seed-bootstrap 95% CI |","|---|---|---|"])
    for key,value in result["paired_effects"].items():
        if key.startswith("progress:"):
            lines.append(f"| {key.split(':',1)[1]} | {value['mean']:.4f} | [{value['ci95'][0]:.4f}, {value['ci95'][1]:.4f}] |")
    lines.extend(["","For exact SPD nested subspaces, removing measurements can only reduce one-step",
      "model progress. Any count advantage must therefore be established on cost,",
      "noise avoidance, or repeated updates. A larger D/A progress gap alone cannot",
      "establish a beneficial count/alignment interaction.","",
      "The adaptive Krylov update agrees with conventional CG to the independently",
      "checked finite-precision tolerance. Its directional advantage is established",
      "prior art; these observations do not demonstrate a new tomography algorithm.","",
      "Family/dimension/gradient results, all 240 candidate/cap summaries, paired",
      "effects, matched-mean-count mixtures, and cost sensitivity are in",
      "[main_summary.json](results/main_summary.json) and",
      "[candidate_summary.json](results/candidate_summary.json).",
      "The main effect intervals are descriptive and are not multiplicity-adjusted",
      "claims of a universal optimizer winner."])
    Path("analysis/monte_carlo.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"cells":result["cells"],"matched_mean_count":result["matched_mean_count"]},indent=2))


if __name__=="__main__":
    make_report()
