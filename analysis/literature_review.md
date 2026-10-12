# Primary-source review and novelty boundary

This is a bounded, problem-oriented review rather than a claim to have read every
paper in every field. Thirty primary papers were downloaded and their extracted
texts hashed in [`literature/retrieval.json`](literature/retrieval.json); source
metadata and primary links are in [`literature/sources.json`](literature/sources.json).
The reading log below records inspected sections in extracted-text line numbers
(the PDFs retain their original pagination). Each note states the relevant
formulation, information returned, target, guarantee or limit, and overlap. The
notes paraphrase; consult the linked source for exact assumptions and theorem
statements. Local cached PDFs and full texts are intentionally excluded from git.

## Optimization, Krylov, and curvature updates

| Source | Formulation, oracle and target | Guarantee or limit; relationship to this study |
|---|---|---|
| [Hestenes & Stiefel (1952), Conjugate Gradients](https://nvlpubs.nist.gov/nistpubs/jres/049/jresv49n6p409_A1b.pdf), `cg52`, §§ algorithm/convergence (text 475–527) | SPD linear system; matrix-vector products; minimize the associated quadratic. | Finite termination in exact arithmetic and classical condition-number convergence. Residual Krylov in this repository is CG in exact arithmetic; the adaptive vector cell is established optimization targeting. |
| [Pearlmutter (1994), Fast Exact Multiplication by the Hessian](https://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf), `hvp94`, derivation/application (130–175) | Automatic differentiation computes `Hv` from reverse/forward differentiation without forming `H`; vector oracle. | Establishes practical Hessian-vector products. It does not make a scalar `vᵀHv` query free: that scalar may still require the same product. |
| [Martens (2010), Deep Learning via Hessian-Free Optimization](https://icml.cc/Conferences/2010/papers/458.pdf), `hf10`, §§ Hessian-free method and damping (111–174) | Damped curvature-vector products inside truncated CG; reduce local loss in neural networks. | Uses inexact inner solves and damping/acceptance machinery. Direct prior art for adaptive directions and stopping in a neural optimization setting. |
| [Hennig & Kiefel (2013), Quasi-Newton Methods: A New Direction](https://www.jmlr.org/papers/volume14/hennig13a/hennig13a.pdf), `quasi13`, methods/results (209–260, 290–347) | Bayesian inference over inverse-Hessian information from secant pairs; optimize a step using posterior curvature. | Posterior depends on prior and observation model; secant observations are gradient differences, not arbitrary HVPs. Bayesian curvature acquisition is established; this project does not claim the general idea as new. |
| [Gower & Richtárik (2017), Randomized Quasi-Newton Updates](https://arxiv.org/pdf/1602.01768), `randqn17`, algorithm and convergence (314–360, 1776–1820) | Randomized matrix-inversion sketch/update, typically using a matrix-vector product with the system matrix; target inverse action. | Linear convergence under stated sampling and positive-definiteness assumptions. Randomized curvature updates already adapt information to inverse reconstruction; objective-specific acquisition must be compared carefully. |
| [Gower, Goldfarb & Richtárik (2016), Stochastic Block BFGS](https://proceedings.mlr.press/v48/gower16.pdf), `blockbfgs16`, algorithm/analysis (185–224) | Block secant information and stochastic Hessian samples; construct curvature approximations for optimization. | Convergence is tied to sampling and regularity assumptions. Multiple curvature directions per update and block costs are prior art. |
| [Gower et al. (2019), RSN: Randomized Subspace Newton](https://proceedings.neurips.cc/paper_files/paper/2019/file/bc6dc48b743dc5d013b1abaebd2faed2-Paper.pdf), `rsn19`, method and theory (85–172, 190–275) | Random subspace Newton update based on Hessian information projected into a sampled subspace; optimize convex objectives. | Expected convergence under the paper's convexity/smoothness and sampling conditions. Direct overlap with subspace curvature updates; the present common Galerkin step and four-cell factorial isolate acquisition alignment and count. |
| [Pilanci & Wainwright (2017), Newton Sketch](https://stanford.edu/~pilanci/papers/PilWai17.pdf), `newtonsketch17`, algorithm/theory (352–396, 812–855) | Sketch Hessian factors or data features to form approximate Newton systems; optimize generalized linear models. | Near-linear iteration costs and local convergence under explicit sketch dimension and regularity conditions. Sketching curvature for optimization is established, although the oracle/cost model differs from direct HVP acquisition. |
| [Lacotte et al. (2021), Adaptive Newton Sketch](https://proceedings.mlr.press/v139/lacotte21a/lacotte21a.pdf), `adaptivesketch21`, adaptive sketch rule/theory (225–274, 448–504) | Increase sketch accuracy based on observed Newton decrement/optimization progress; target efficient Newton steps. | Convergence and complexity depend on effective dimension and assumptions in the paper. Adaptive curvature effort is explicit prior art; count adaptation alone is not a novelty claim. |
| [Frangella, Tropp & Udell (2023), Randomized Nyström Preconditioning](https://tropp.caltech.edu/papers/FTU23-Randomized-Nystrom-SIMAX.pdf), `nystrom23`, construction and guarantees (125–191, 355–400, 740–785) | Matrix-vector products with a PSD operator create a low-rank Nyström preconditioner; accelerate iterative solves. | Spectral approximation and convergence bounds under PSD/rank assumptions. Our reference implementation follows this mechanism; low-rank HVP preconditioning is prior art. |
| [Frangella et al. (2024), SketchySGD](https://arxiv.org/pdf/2211.08597), `sketchysgd24`, estimator and analysis (260–310, 492–529) | Randomized sketches estimate and apply curvature information during stochastic optimization. | Guarantees depend on sampling, smoothness, and sketch/error controls. It reinforces that scalable stochastic curvature estimation is a substantial neighboring field. |
| [Bollapragada, Byrd & Nocedal (2019), Exact and Inexact Subsampled Newton](https://optimization-online.org/wp-content/uploads/2016/09/5653.pdf), `subsample19`, algorithms/complexity (786–828) | Sampled Hessian or Hessian-vector products with inexact Newton solves; target nonlinear optimization. | Convergence depends on sampling accuracy, forcing conditions, and regularity. Inexact solve accuracy and sampling costs are existing tools, not unique contributions here. |
| [Eisenstat & Walker (1996), Choosing the Forcing Terms](https://users.wpi.edu/~walker/Papers/forcing_terms%2CSISC_17%2C1996%2C16-32.pdf), `forcing96`, forcing rules/theory (193–221, 282–308) | Choose an inner residual tolerance for inexact Newton from outer iteration progress. | Local convergence results under stated smoothness and forcing conditions. The residual stopping rules in our experiments are conventional forcing criteria; a new economic interpretation would require separate evidence. |
| [MINRES negative-curvature analysis (2022)](https://arxiv.org/pdf/2206.05732), `minres22`, recurrence and monotonicity results (380–425) | Symmetric indefinite matrix-vector products solve least-squares/symmetric systems and expose spectral behavior. | Results concern Krylov recurrences and monotonicity under stated hypotheses. Negative-curvature discovery and indefinite Krylov methods are established; our saddle example is a small illustration only. |
| [Gupta, Koren & Singer (2018), Shampoo](https://proceedings.mlr.press/v80/gupta18a.html), `shampoo18`, update and convergence (61–85) | Structured matrix/tensor second moments precondition stochastic gradients. | Scalable structured state with convergence under convex stochastic assumptions. It changes the curvature/statistics representation and is not a direct HVP-direction policy. |
| [Martens & Grosse (2015), K-FAC](https://proceedings.mlr.press/v37/martens15.html), `kfac15`, factorization/algorithm (144–182, 277–312) | Kronecker-factor approximations to Fisher/Gauss–Newton curvature for neural networks. | Practical structured preconditioning with approximation assumptions. Relevant large-model feasibility prior art; full dense covariance or PCA directions from our toy study do not scale as K-FAC does. |
| [Yao et al. (2021), AdaHessian](https://arxiv.org/pdf/2006.00719), `adahessian21`, estimator and experiments (520–584) | Stochastic diagonal Hessian estimation using randomized probes; adapt per-coordinate learning rates. | Diagonal approximation trades curvature fidelity for cost. Scalar/trace-style random probes and optimizer deployment are existing approaches; the objective differs from a Galerkin Newton step. |
| [Liu et al. (2023), Sophia](https://arxiv.org/pdf/2305.14342), `sophia23`, estimator/update (122–168) | Lightweight diagonal curvature estimate and clipped stochastic update for language-model pretraining. | Empirical scaling results and estimator-specific analysis; it does not reconstruct a full inverse Hessian. Directly relevant deployment evidence, but not evidence for our HVP-based policy. |
| [Vyas et al. (2024), SOAP](https://arxiv.org/pdf/2409.11321), `soap24`, algorithm/experiments (245–286) | Shampoo-derived preconditioner with Adam-style moment updates in an estimated eigenbasis. | Practical structured optimizer with empirical claims and approximation costs. Directional/structured optimizer prior art; no same oracle or objective comparison. |
| [Jahani et al. (2021), Doubly Adaptive Scaled Algorithm](https://arxiv.org/pdf/2109.05198), `dan21`, OASIS algorithm/analysis (328–382) | Doubly adaptive scaling uses second-order information and stochastic updates. | Guarantees are tied to its assumptions and algorithm. It is a meaningful adaptive optimizer comparison, not an oracle-identical acquisition method. |

## Identification, randomized linear algebra, and experimental design

| Source | Formulation, oracle and target | Guarantee or limit; relationship to this study |
|---|---|---|
| [Kueng, Rauhut & Terstiege (2017), Low Rank Matrix Recovery from Rank-One Measurements](https://www.mathc.rwth-aachen.de/~rauhut/files/LowRankMatrixRecoveryKRT.pdf), `matrixrank17`, measurement model/recovery theory (125–220, 212–250) | Scalar rank-one measurements `⟨vvᵀ,H⟩=vᵀHv`; identify a structured low-rank matrix. | Recovery requires low-rank structure and measurement conditions; generic symmetric recovery has `n(n+1)/2` degrees of freedom. It does not imply optimization-relevant inverse action can be recovered cheaply. |
| [Halko, Martinsson & Tropp (2011), Finding Structure with Randomness](https://arxiv.org/pdf/0909.4061), `randsvd11`, randomized range finder/error bounds (407–442) | Matrix-vector or matrix-block products construct approximate ranges and decompositions. | Probabilistic spectral/Frobenius approximation bounds under stated sampling and oversampling. These target matrix reconstruction; gradient-weighted step accuracy is a different loss. |
| [Meyer et al. (2021), Hutch++](https://arxiv.org/pdf/2010.09649), `hutchpp21`, estimator/variance theorem (140–168, 187–243) | Matrix-vector products estimate trace with randomized range plus residual probing. | Improved trace-estimation sample complexity for suitable PSD matrices. Trace accuracy does not itself control `H⁻¹g`; scalar curvature measurements and vector products remain different information models. |
| [Persson, Cortinovis & Kressner (2022), Improved Hutch++ variants](https://arxiv.org/pdf/2109.10659), `adaptivehutch22`, adaptive estimator analysis (489–535) | Adaptive randomized trace estimation using matrix-vector products. | Variance and sample guarantees target trace estimation. It is a useful adaptive-estimation comparison, but not a Newton-step objective. |
| [Krause & Guestrin (2005), Near-optimal Nonmyopic Value of Information](https://www.cs.cmu.edu/~guestrin/Publications/UAI2005/uai2005.pdf), `value05`, objective and theorem (220–262) | Sequential observations maximize expected information/value in graphical models. | Approximation results rely on submodularity or related structure. Our exact two-direction counterexample shows the optimization-gain set function need not be submodular, so generic greedy guarantees do not transfer. |
| [Castro (2014), Adaptive Sensing Performance Lower Bounds](https://arxiv.org/pdf/1206.0648), `castro14`, adaptive lower bounds (225–251) | Sequential scalar measurements for sparse detection/support recovery. | Lower bounds show limits of adaptivity for those sparse statistical tasks. The model is not Hessian recovery; it motivates distinguishing the scalar identifiability question without transferring a bound. |
| [Attia, Alexanderian & Saibaba (2018), Goal-Oriented Optimal Design](https://arxiv.org/pdf/1802.06517), `goaldesign18`, goal-oriented design (480–507) | Choose linear observations to reduce uncertainty in a specified quantity of interest. | Bayesian linear inverse-problem criteria target a goal rather than full parameter entropy. This is direct conceptual prior art for inverse-step uncertainty; our nonlinear goal uses a local Jacobian approximation and no new design guarantee. |

## Probabilistic linear solvers and limits of interpretation

| Source | Formulation, oracle and target | Guarantee or limit; relationship to this study |
|---|---|---|
| [Bartels et al. (2019), Probabilistic Linear Solvers: A Unifying View](https://arxiv.org/pdf/1810.03398), `problinsolve19`, observation/update view (540–587) | Bayesian inference over a linear-system solution from matrix-vector and related observations. | Posterior depends on prior, likelihood, and query design; exactness is conditional on that model. HVP-driven probabilistic inference is established. |
| [Wenger et al. (2020), Probabilistic Linear Solvers for Machine Learning](https://arxiv.org/pdf/2010.09691), `problinsolve20`, methods/ML applications (280–327, 430–474) | Apply probabilistic linear solves to machine-learning computations, with uncertainty over numerical solutions. | Practical benefit depends on calibrated uncertainty and downstream use. Our scalar Gaussian study is explicitly finite-dimensional and locally linearized, not a scalable solver claim. |
| [Powell (2006), NEWUOA](https://optimization-online.org/wp-content/uploads/2005/01/1045.pdf), `newuoa06`, interpolation geometry/model updating (95–142) | Scalar function values build and update a local quadratic model without derivatives. | Trust-region model accuracy depends on poised interpolation sets and smoothness. It is derivative-free function-value access, unlike direct scalar Hessian measurements; included as an oracle-boundary comparison. |

## Search coverage, open gaps, and novelty statement

Searches covered primary sources in classical and inexact Newton/CG, HVPs,
randomized and adaptive sketching, Nyström preconditioning, quasi-Newton and
probabilistic linear solvers, scalar rank-one matrix recovery, randomized
decomposition/trace estimation, goal-oriented design, information acquisition,
nonconvex negative curvature, and structured neural optimizers. Queries included
cross-field terms such as *adaptive Hessian-vector directions*, *optimization
targeted curvature*, *goal-oriented matrix-vector design*, *Hessian tomography*,
and the named method families above. Thirty retrieved primary texts have stable
SHA-256 records. The reading locators are in the table and are section-level
checks of extracted full text, not a page-by-page audit of every source.

Recent 2026 OpenReview records surfaced in search, but direct paper access was
blocked and the available search snippets did not establish reliable titles or
claims; they remain unresolved leads. A recent adaptive trace-estimation result
also surfaced but is peripheral and was not used to support any claim. The
search cannot certify exhaustiveness, especially for papers published after the
retrieval date, workshops, non-English work, and adjacent active-learning or
probabilistic-numerics literature.

**Conservative novelty assessment:** no novelty claim is established. Adaptive
Krylov directions are CG; residual tolerances are inexact Newton forcing rules;
adaptive sketch size and Nyström preconditioning have direct prior art; Bayesian
goal-oriented design already selects measurements for a quantity of interest.
The potentially useful synthesis is to compare scalar versus vector curvature
oracles and fixed/adaptive direction and count choices under a shared
optimization-value objective, with explicit information and compute accounting.
That is a research framing and benchmark contribution at most, pending a wider
review, a more scalable goal estimator, and evaluation on actual training
systems. The finite experiments here support no language-model superiority or
general algorithmic novelty claim.
