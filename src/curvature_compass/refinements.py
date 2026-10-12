"""Independent mechanism studies registered in refinement_protocol.md."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
import argparse
import csv
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import brentq
from .core import FAMILIES, VectorOracle, acquire, cg, fixed_stream, orthogonalize, problem
from .study import ROOT, seed_for


def nystrom_preconditioner(oracle,n,rank,seed,ridge=1.0):
    Q=fixed_stream(n,"haar",seed,size=rank)
    Y=np.column_stack([oracle(v)-ridge*v for v in Q.T])
    W=(Q.T@Y+Y.T@Q)/2
    values,vectors=np.linalg.eigh(W)
    factor=Y@(vectors/np.sqrt(np.maximum(values,1e-12)))
    U,sv,_=np.linalg.svd(factor,full_matrices=False)
    eigen=sv**2
    # FTU23 equation (1.3): P^-1 equals identity off the sketch range.
    scale=(eigen[-1]+ridge)/(eigen+ridge)
    return lambda r:r+U@((scale-1)*(U.T@r))


def pcg(oracle,g,cap,preconditioner,tolerance=1e-10):
    x=np.zeros_like(g); r=-g.copy(); z=preconditioner(r); p=z.copy(); rz=r@z
    for _ in range(cap):
        Ap=oracle(p)
        denominator=p@Ap
        if denominator<=0:
            break
        alpha=rz/denominator
        x+=alpha*p; r-=alpha*Ap
        if np.linalg.norm(r)<=max(tolerance*np.linalg.norm(g),1e-13):
            break
        z=preconditioner(r); next_rz=r@z
        p=z+(next_rz/rz)*p; rz=next_rz
    return x


def reference_study():
    records=[]
    for seed in range(3000,3032):
        for family in FAMILIES:
            p=problem(family,48,seed_for(seed,family,48))
            Fstar=.5*p.g@np.linalg.solve(p.H,p.g)
            for budget in (8,16):
                for rank in (0,2,4,8):
                    if rank>=budget:
                        continue
                    oracle=VectorOracle(lambda v:p.H@v,48)
                    started=time.perf_counter()
                    if rank:
                        P=nystrom_preconditioner(oracle,48,rank,seed)
                        s=pcg(oracle,p.g,budget-rank,P,tolerance=0)
                    else:
                        s=cg(oracle,p.g,budget)
                    progress=(-p.g@s-.5*s@p.H@s)/Fstar
                    records.append({"kind":"reference","seed":seed,"family":family,
                        "budget":budget,"rank":rank,"count":oracle.calls,
                        "progress":float(progress),"seconds":time.perf_counter()-started})
    return records


def iterative_study():
    selection=json.loads((ROOT/"selection.json").read_text())
    priors=dict(np.load(ROOT/"offline_priors.npz"))
    records=[]
    methods=list("ABCD")+["CG8","CG_residual","steepest","persistent_CG","persistent_Nystrom_PCG","LBFGS5"]
    for seed in range(3000,3032):
        for family in FAMILIES:
            H=problem(family,48,seed_for(seed,family,48)).H
            initial=np.random.default_rng(seed*10000+FAMILIES.index(family)).normal(size=48)
            F0=.5*initial@H@initial
            for budget_type,budget in (("hvp",128),("work",256)):
                for method in methods:
                    if method=="LBFGS5" and budget_type=="hvp":
                        continue
                    x=initial.copy(); hvps=grads=losses=steps=0; memory=[]; previous=None
                    current_loss=float(F0)
                    if method=="LBFGS5":
                        losses=1  # its initial function value is used by Armijo
                    start=time.perf_counter()
                    while (hvps if budget_type=="hvp" else hvps+grads+losses)<budget:
                        available=budget-(hvps if budget_type=="hvp" else hvps+grads+losses)
                        if budget_type=="work" and available<2:
                            break
                        g=H@x; grads+=1
                        if np.linalg.norm(g)<1e-10*np.linalg.norm(H@initial):
                            break
                        cap=min(8,int(available)-(1 if budget_type=="work" else 0))
                        oracle=VectorOracle(lambda v:H@v,48)
                        if method.startswith("persistent"):
                            cap=int(available)-(1 if budget_type=="work" else 0)
                            if method=="persistent_Nystrom_PCG" and cap>4:
                                P=nystrom_preconditioner(oracle,48,4,seed)
                                step=pcg(oracle,g,cap-4,P)
                            else:
                                step=cg(oracle,g,cap,1e-10)
                        elif method in selection["configurations"]:
                            config=selection["configurations"][method]
                            # Offset is determined by OUTER STEP, independent of
                            # measured count, g, Hessian and observed history.
                            prior=np.roll(priors["48"],-(steps*8)%48,axis=1)
                            path=acquire(oracle,g,config["alignment"],cap,seed+steps*7919,
                                config["stopping"],config["tolerance"],prior=prior)
                            step=path[-1]["step"] if path else np.zeros(48)
                        elif method in ("CG8","CG_residual"):
                            step=cg(oracle,g,cap,.7 if method=="CG_residual" else 0)
                        elif method=="steepest":
                            Hg=oracle(g); step=-(g@g)/(g@Hg)*g
                        else:
                            if previous is not None:
                                s,old_g=previous; y=g-old_g
                                if s@y>1e-12*np.linalg.norm(s)*np.linalg.norm(y):
                                    memory.append((s,y,1/(s@y))); memory=memory[-5:]
                            q=g.copy(); alphas=[]
                            for s,y,rho in reversed(memory):
                                alpha=rho*(s@q); alphas.append(alpha); q-=alpha*y
                            scale=(memory[-1][0]@memory[-1][1])/(memory[-1][1]@memory[-1][1]) if memory else 1.0
                            z=scale*q
                            for (s,y,rho),alpha in zip(memory,reversed(alphas)):
                                z+=s*(alpha-rho*(y@z))
                            direction=-z
                            if g@direction>=0:
                                direction=-g
                            alpha=1.; step=np.zeros(48)
                            for _ in range(32):
                                if hvps+grads+losses>=budget:
                                    break
                                trial=x+alpha*direction; ftrial=float(.5*trial@H@trial); losses+=1
                                if ftrial<=current_loss+1e-4*alpha*(g@direction):
                                    step=alpha*direction; current_loss=ftrial; break
                                alpha*=.5
                            previous=(step.copy(),g.copy())
                        hvps+=oracle.calls; x+=step; steps+=1
                        actual=float(.5*x@H@x)
                        if not np.isfinite(actual) or actual>F0*(1+1e-7):
                            raise AssertionError((method,family,actual,F0))
                        if method!="LBFGS5":
                            current_loss=actual  # evaluator only, not policy input
                        records.append({"kind":"iterative","seed":seed,"family":family,
                            "method":method,"budget_type":budget_type,"step":steps,
                            "hvps":hvps,"gradients":grads,"loss_queries":losses,
                            "work":hvps+grads+losses,"relative_loss":actual/F0,
                            "seconds":time.perf_counter()-start})
                        if np.linalg.norm(step)<1e-15:
                            break
    return records


class RecordedOracle(VectorOracle):
    def __init__(self,H,noise=0.,seed=0):
        super().__init__(lambda v:H@v,len(H),seed=seed)
        self.matrix=H; self.radius=noise; self.V=[]; self.Y=[]

    def __call__(self,v):
        y=super().__call__(v)
        if self.radius:
            z=self.rng.normal(size=len(v)); y=y+self.radius*z/np.linalg.norm(z)
        self.V.append(v.copy()); self.Y.append(y.copy())
        return y


def observed_residual(oracle,g,step,count):
    V=np.column_stack(oracle.V[:count]); Y=np.column_stack(oracle.Y[:count])
    return g+Y@(V.T@step)


def noise_study():
    priors=dict(np.load(ROOT/"offline_priors.npz")); records=[]
    for seed in range(4000,4016):
        for n in (16,48):
            for family in ("flat","rotated","lowrank","needle"):
                p=problem(family,n,seed_for(seed,family,n))
                Fstar=.5*p.g@np.linalg.solve(p.H,p.g)
                rng=np.random.default_rng(seed_for(seed,family,n)+91)
                E=rng.normal(size=(n,n)); E=(E+E.T)/2; E/=np.linalg.norm(E,2)
                z=rng.normal(size=n); z/=np.linalg.norm(z)
                for amplitude in (0.,.05,.25,1.5):
                    for gradient_noise in (0.,.1):
                        gobs=p.g+gradient_noise*z
                        for mode in ("coherent","fresh"):
                            Hobs=p.H+amplitude*E if mode=="coherent" else p.H
                            for alignment in ("prior","krylov"):
                                oracle=RecordedOracle(Hobs,amplitude if mode=="fresh" else 0,seed)
                                path=acquire(oracle,gobs,alignment,8,seed,prior=priors[str(n)],safety_floor=.1)
                                for stop in ("observed_residual","robust_residual"):
                                    selected=None
                                    for row in path:
                                        s=row["step"].copy()
                                        # Common trust cap limits catastrophic noisy inverses.
                                        s*=min(1.,1/max(np.linalg.norm(s),1e-15))
                                        ro=observed_residual(oracle,gobs,s,row["count"])
                                        eps=amplitude*(np.sqrt(row["count"]) if mode=="fresh" else 1.)
                                        bound=np.linalg.norm(ro)+eps*np.linalg.norm(s)+gradient_noise
                                        selected=(row,s,ro,eps,bound)
                                        criterion=np.linalg.norm(ro) if stop=="observed_residual" else bound
                                        if criterion<=.7*np.linalg.norm(gobs):
                                            break
                                    row,s,ro,eps,bound=selected
                                    actual=float(-p.g@s-.5*s@p.H@s)
                                    gap=Fstar-actual
                                    observed_certificate=float(ro@ro/2)
                                    robust_certificate=float(bound**2/2)
                                    assert gap<=robust_certificate+1e-8
                                    # Evaluation audit: one extra TRUE HVP, separately charged.
                                    validation=VectorOracle(lambda v:p.H@v,n)
                                    true_r=p.g+validation(s)
                                    records.append({"kind":"noise","seed":seed,"family":family,"dimension":n,
                                        "amplitude":amplitude,"gradient_noise":gradient_noise,"noise_model":mode,
                                        "alignment":alignment,"stopping":stop,"count":row["count"],
                                        "validation_hvps":validation.calls,"progress":actual/Fstar,
                                        "observed_certificate":observed_certificate,"robust_certificate":robust_certificate,
                                        "true_gap":float(gap),"true_residual":float(np.linalg.norm(true_r)),
                                        "observed_certificate_failure":gap>observed_certificate+1e-8})
    return records


def drift_study():
    records=[]
    for seed in range(4000,4016):
        p=problem("rotated",48,seed*1000)
        for angle in (0.,.05,.25,.75):
            R=np.eye(48); c,s=np.cos(angle),np.sin(angle)
            R[0,0]=R[-1,-1]=c; R[0,-1]=-s; R[-1,0]=s
            Hcurrent=R@p.H@R.T
            Fstar=.5*p.g@np.linalg.solve(Hcurrent,p.g)
            epsilon=np.linalg.norm(Hcurrent-p.H,2)
            for alignment in ("haar","krylov"):
                oracle=RecordedOracle(p.H)
                path=acquire(oracle,p.g,alignment,8,seed,stopping="residual",tolerance=.7)
                row=path[-1]; step=row["step"]
                ro=observed_residual(oracle,p.g,step,row["count"])
                gap=Fstar-(-p.g@step-.5*step@Hcurrent@step)
                bound=(np.linalg.norm(ro)+epsilon*np.linalg.norm(step))**2/2
                assert gap<=bound+1e-8
                records.append({"kind":"drift","seed":seed,"angle":angle,"alignment":alignment,
                    "count":row["count"],"progress":float(1-gap/Fstar),"true_gap":float(gap),
                    "observed_certificate":float(ro@ro/2),"robust_certificate":float(bound),
                    "observed_certificate_failure":bool(gap>ro@ro/2+1e-8)})
    return records


def trust_region(G,g,radius):
    """Global minimizer of a small symmetric trust-region quadratic, hard case included."""
    eigen,U=np.linalg.eigh((G+G.T)/2); w=U.T@g
    lower=max(0.,-eigen[0]); denominator=eigen+lower
    singular=denominator<1e-10
    c=np.zeros_like(w); c[~singular]=-w[~singular]/denominator[~singular]
    if not np.any(np.abs(w[singular])>1e-10) and np.linalg.norm(c)<=radius:
        if lower>0:
            c[np.argmin(eigen)]=np.sqrt(max(0.,radius**2-c@c))
        return U@c
    def norm_at(shift):
        return np.linalg.norm(w/(eigen+shift))
    low=lower+1e-12; high=max(1.,lower+1.)
    while norm_at(high)>radius:
        high=2*high+1
    shift=brentq(lambda t:norm_at(t)-radius,low,high,xtol=1e-12)
    return -U@(w/(eigen+shift))


def nonconvex_study():
    records=[]; n=16
    for seed in range(5000,5032):
        rng=np.random.default_rng(seed)
        U=np.linalg.qr(rng.normal(size=(n,n)))[0]
        A=(U*np.r_[-2.,-1.,np.geomspace(1,30,n-2)])@U.T
        x=rng.normal(size=n); x*=.3/np.linalg.norm(x)
        g=A@x+(x@x)*x
        H=A+(x@x)*np.eye(n)+2*np.outer(x,x)
        def loss(z): return float(.5*z@A@z+.25*(z@z)**2)
        f0=loss(x)
        for alignment in ("haar","coordinate","krylov","greedy_coordinate"):
            oracle=RecordedOracle(H)
            path=acquire(oracle,g,alignment,8,seed,safety_floor=.1)
            Q=np.column_stack(oracle.V); G=Q.T@H@Q; rhs=Q.T@g
            for reconstruction in ("clipped","trust_region"):
                if reconstruction=="clipped":
                    step=path[-1]["step"].copy(); step*=min(1.,.5/max(np.linalg.norm(step),1e-15))
                else:
                    step=Q@trust_region(G,rhs,.5)
                raw=loss(x+step); alpha=1.; accepted=False; trials=0
                for _ in range(20):
                    trials+=1; ftrial=loss(x+alpha*step)
                    if ftrial<=f0+1e-4*alpha*min(g@step,0.):
                        accepted=True; break
                    alpha*=.5
                if not accepted:
                    alpha=0.; ftrial=f0
                assert ftrial<=f0+1e-9
                records.append({"kind":"nonconvex","seed":seed,"alignment":alignment,
                    "reconstruction":reconstruction,"hvps":oracle.calls,"loss_queries":trials,
                    "raw_loss_change":raw-f0,"accepted_loss_change":ftrial-f0,
                    "accepted":accepted,"step_scale":alpha})
        # A stationary saddle defeats a gradient-only seed. Random-start Lanczos
        # reveals both eigenspaces of I-3uuᵀ with two exact HVPs.
        Asaddle=np.eye(n)-3*np.outer(U[:,0],U[:,0])
        oracle=VectorOracle(lambda v:Asaddle@v,n)
        q=rng.normal(size=n); q/=np.linalg.norm(q); basis=[]; actions=[]
        for _ in range(2):
            q=orthogonalize(q,np.column_stack(basis) if basis else np.empty((n,0)))
            if q is None: break
            y=oracle(q); basis.append(q); actions.append(y); q=y
        Q=np.column_stack(basis); Y=np.column_stack(actions)
        step=Q@trust_region(Q.T@Y,np.zeros(len(basis)),.5)
        decrease=-float(.5*step@Asaddle@step+.25*(step@step)**2)
        assert decrease>.1
        records.append({"kind":"saddle","seed":seed,"gradient_only_decrease":0.,
            "exploration_decrease":decrease,"exploration_hvps":oracle.calls})
    return records


def accuracy_study():
    priors=dict(np.load(ROOT/"offline_priors.npz")); records=[]
    for seed in range(6000,6032):
        for family in FAMILIES:
            for regime in ("isotropic","error","low","high"):
                for condition in (10.,1000.,100000.):
                    p=problem(family,48,seed_for(seed,family,48),regime,condition)
                    Fstar=.5*p.g@np.linalg.solve(p.H,p.g)
                    for alignment in ("prior","krylov"):
                        path=acquire(VectorOracle(lambda v:p.H@v,48),p.g,alignment,16,seed,prior=priors["48"])
                        for rule in ("residual_0.7","residual_0.1","objective_certificate"):
                            selected=path[-1]; met=False
                            for row in path:
                                criterion=(row["certificate"]<=row["decrease_model"]/9 if rule=="objective_certificate"
                                     else row["residual_norm"]<=(.7 if rule.endswith("0.7") else .1))
                                if criterion:
                                    selected=row; met=True; break
                            s=selected["step"]; progress=float((-p.g@s-.5*s@p.H@s)/Fstar)
                            if rule=="objective_certificate" and met:
                                assert progress>=.9-1e-7
                            records.append({"kind":"accuracy","seed":seed,"family":family,"regime":regime,
                                "condition_requested":condition,"alignment":alignment,"rule":rule,
                                "count":selected["count"],"progress":progress,"criterion_met":met,
                                "actual_target_met":progress>=.9-1e-7})
    return records


def symmetric_basis(n):
    basis=[]
    for i in range(n):
        B=np.zeros((n,n)); B[i,i]=1.; basis.append(B)
    for i in range(n):
        for j in range(i+1,n):
            B=np.zeros((n,n)); B[i,j]=B[j,i]=1/np.sqrt(2); basis.append(B)
    return np.array(basis)


def scalar_study():
    records=[]
    for seed in range(7000,7064):
        for n in (4,8):
            rng=np.random.default_rng(seed+n); basis=symmetric_basis(n); d=len(basis)
            pool=np.vstack((np.eye(n),np.random.default_rng(seed+61).normal(size=(128,n))))
            pool/=np.linalg.norm(pool,axis=1)[:,None]
            features=np.einsum("bi,kij,bj->bk",pool,basis,pool)
            sigma=.02; prior_variance=.2**2
            # Entropy alignment is completely preplanned: Gaussian posterior
            # covariance depends only on designs, never observed values or g.
            covariance=prior_variance*np.eye(d); planned=[]
            for _ in range(16):
                products=features@covariance
                variances=np.sum(products*features,axis=1)
                chosen=int(np.argmax(variances)); planned.append(chosen)
                u=products[chosen]
                covariance-=np.outer(u,u)/(sigma**2+variances[chosen])
            latent=rng.normal(size=d)*.2
            g=rng.normal(size=n) if seed%2 else np.eye(n)[int(rng.integers(n))]
            g/=np.linalg.norm(g)
            noise=rng.normal(size=16)*sigma
            for true_scale in (1.,3.):
                H=5*np.eye(n)+np.einsum("k,kij->ij",true_scale*latent,basis)
                if np.linalg.eigvalsh(H).min()<=1:
                    raise AssertionError("Generated scalar prior left declared SPD population")
                truth=-np.linalg.solve(H,g)
                for design in ("entropy_fixed","inverse_goal_adaptive"):
                    mean=np.zeros(d); covariance=prior_variance*np.eye(d); path=[]
                    initial_risk=None
                    for index in range(16):
                        Hmean=5*np.eye(n)+np.einsum("k,kij->ij",mean,basis)
                        eigen,U=np.linalg.eigh(Hmean); inverse=(U/np.maximum(eigen,1.))@U.T
                        step=-inverse@g
                        J=np.column_stack([-inverse@B@step for B in basis])
                        risk=float(np.trace(J@covariance@J.T))
                        if initial_risk is None: initial_risk=risk
                        products=features@covariance
                        variances=np.sum(products*features,axis=1)
                        if design=="entropy_fixed":
                            chosen=planned[index]
                        else:
                            scores=np.sum((J@products.T)**2,axis=0)/(sigma**2+variances)
                            chosen=int(np.argmax(scores))
                        phi=features[chosen]; u=products[chosen]; denom=sigma**2+variances[chosen]
                        observed=float(pool[chosen]@H@pool[chosen]+noise[index])
                        mean+=u*(observed-5-phi@mean)/denom
                        covariance-=np.outer(u,u)/denom; covariance=(covariance+covariance.T)/2
                        Hmean=5*np.eye(n)+np.einsum("k,kij->ij",mean,basis)
                        eigen,U=np.linalg.eigh(Hmean); inverse=(U/np.maximum(eigen,1.))@U.T
                        step=-inverse@g
                        J=np.column_stack([-inverse@B@step for B in basis])
                        path.append({"count":index+1,"step_error_squared":float(np.sum((step-truth)**2)),
                            "matrix_error_squared":float(np.sum((Hmean-H)**2)),
                            "linearized_step_risk":float(np.trace(J@covariance@J.T)),
                            "matrix_variance":float(np.trace(covariance)),"initial_risk":initial_risk})
                    for cap in (2,8,16):
                        for counting in ("fixed","adaptive"):
                            selected=path[cap-1]
                            if counting=="adaptive":
                                for row in path[:cap]:
                                    if row["linearized_step_risk"]<=.4*initial_risk:
                                        selected=row; break
                            label=("A" if design=="entropy_fixed" else "B") if counting=="fixed" else ("C" if design=="entropy_fixed" else "D")
                            records.append({"kind":"scalar_design","seed":seed,"dimension":n,
                                "true_prior_scale":true_scale,"design":design,"counting":counting,
                                "cell":label,"cap":cap,**selected})
            # Separate exact full-matrix rank/recovery study; fixed coordinate
            # scalars remain rank-n even if repeatedly queried d times.
            H=np.eye(n)+rng.normal(size=(n,n))@rng.normal(size=(n,n)).T
            H=(H+H.T)/2
            h=np.einsum("kij,ij->k",basis,H)
            polarization=[np.eye(n)[i] for i in range(n)]
            polarization.extend((np.eye(n)[i]+np.eye(n)[j])/np.sqrt(2) for i in range(n) for j in range(i+1,n))
            designs={"coordinate_repeated":np.resize(np.eye(n),(d,n)),"polarization":np.array(polarization),
                     "gaussian":pool[n:n+d]}
            for name,V in designs.items():
                M=np.einsum("bi,kij,bj->bk",V,basis,V)
                estimate=np.linalg.lstsq(M,M@h,rcond=None)[0]
                records.append({"kind":"scalar_rank","seed":seed,"dimension":n,"design":name,
                    "count":len(V),"rank":int(np.linalg.matrix_rank(M)),"unknowns":d,
                    "relative_reconstruction_error":float(np.linalg.norm(estimate-h)/np.linalg.norm(h))})
                if name=="polarization":
                    assert np.linalg.matrix_rank(M)==d and np.allclose(estimate,h)
    return records


def scaling_study():
    records=[]
    for n in (64,512,4096):
        for seed in range(8000,8024):
            rng=np.random.default_rng(seed)
            U=np.linalg.qr(rng.normal(size=(n,8)))[0]; eigen=np.geomspace(10,1000,8)
            def matvec(v): return v+U@(eigen*(U.T@v))
            g=rng.normal(size=n); g/=np.linalg.norm(g)
            for alignment in ("haar","krylov"):
                for cap in (2,8,16):
                    oracle=VectorOracle(matvec,n)
                    started=time.perf_counter(); path=acquire(oracle,g,alignment,cap,seed)
                    seconds=time.perf_counter()-started
                    k=path[-1]["count"]
                    records.append({"kind":"scaling","seed":seed,"dimension":n,"alignment":alignment,
                        "cap":cap,"count":k,"seconds":seconds,"hvp_seconds":oracle.seconds,
                        "overhead_seconds":seconds-oracle.seconds,
                        "minimum_state_bytes":int(8*(2*n*k+k*k))})
    return records


STUDIES={"reference":reference_study,"iterative":iterative_study,"noise":noise_study,
         "drift":drift_study,"nonconvex":nonconvex_study,"accuracy":accuracy_study,
         "scalar":scalar_study,"scaling":scaling_study}


def run(study):
    start=time.perf_counter(); records=STUDIES[study]()
    path=ROOT/(study+".jsonl.gz")
    with gzip.open(path,"wt") as stream:
        for row in records:
            stream.write(json.dumps(row,default=lambda x:bool(x) if isinstance(x,np.bool_) else float(x))+"\n")
    manifest={"study":study,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seconds":time.perf_counter()-start,"records":len(records),
        "raw_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("src/curvature_compass").glob("*.py")},
        "blas_threads":os.environ["OPENBLAS_NUM_THREADS"],"status":"completed"}
    (ROOT/(study+"_manifest.json")).write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:v for k,v in manifest.items() if k!="source_sha256"}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("study",choices=list(STUDIES)+["all"])
    args=parser.parse_args()
    for name in (STUDIES if args.study=="all" else [args.study]):
        run(name)
