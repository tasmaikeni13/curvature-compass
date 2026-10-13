# Research Mission: Adaptive Hessian Tomography for Neural Network Optimization

You are an autonomous mathematical research, scientific computing, formal verification, and deep learning engineering agent.

Your mission is to investigate, develop, formally analyze, implement, and experimentally validate an optimization framework based on **adaptive Hessian tomography**.

Approach this as an original scientific research project, not merely an engineering task. Your responsibility is to discover the strongest justified approaches through literature review, mathematical reasoning, statistical analysis, simulation, experimentation, and rigorous benchmarking.

Do not assume the proposed idea is novel, correct, or superior. Establish what is already known, identify defensible research gaps, develop competing solutions, and let theoretical and empirical evidence determine which survive.

You have substantial autonomy over the research methodology, mathematical tools, algorithms, experimental designs, software architecture, numerical techniques, and engineering decisions.

**Do not begin with a predetermined solution. Investigate before designing, analyze before implementing at scale, and validate before making claims.**

---

# Core Research Idea

The fundamental inspiration is tomography: reconstructing useful higher-dimensional information from lower-dimensional directional observations or "shadows."

For neural-network optimization, consider the Hessian of a loss function:

\[
H(\theta)=\nabla^2_\theta L(\theta)
\]

Rather than constructing the complete Hessian, investigate whether a carefully selected and potentially adaptive collection of directional measurements can reveal enough curvature information to generate effective parameter updates.

The goal is not necessarily to reconstruct the Hessian accurately. The goal is to acquire **the most useful curvature information for optimization, at the lowest justified computational cost**.

The central research questions are:

1. **How many measurement vectors are needed?** Determine how the optimal measurement budget depends on the optimization state, curvature structure, uncertainty, dimensionality, and computational cost.

2. **How should those vectors be positioned or aligned?** Determine which directions provide the most valuable information for constructing an effective update.

3. **Should the number of vectors change adaptively?** Investigate policies that allocate additional measurements only when their expected benefit justifies their cost.

4. **Should vector alignment change adaptively?** Investigate whether directions selected using previously acquired information outperform fixed, random, or predetermined directions.

5. **How should both decisions interact?** Investigate jointly adaptive rules that determine both which measurement to acquire next and whether further measurements are necessary.

6. **What constitutes useful information?** Critically distinguish accurate Hessian reconstruction, spectral information, numerical conditioning, uncertainty reduction, optimization progress, and actual training efficiency.

7. **When does adaptive tomography fail?** Investigate failure modes, overhead, numerical instability, sensitivity to noise, changing curvature, and situations in which simpler optimizers are preferable.

The research must distinguish scalar directional curvature observations, vector-valued Hessian measurements, and other legitimate measurement models. Their information content, identifiability, costs, and mathematical guarantees must not be conflated.

Do not restrict the research to a literal tomography construction if a more general mathematical formulation proves more useful. Preserve the scientific intent while allowing the evidence to determine the appropriate formalism.

---

# Four Experimental Configurations

The research revolves around a controlled comparison of four configurations:

| Configuration | Number of vectors | Vector alignment |
|---|---|---|
| A | Fixed | Fixed |
| B | Fixed | Adaptive |
| C | Adaptive | Fixed |
| D | Adaptive | Adaptive |

These are the four principal experimental configurations throughout the project.

Define "fixed" and "adaptive" precisely, accounting for changes across measurements and training steps. Ensure the four configurations remain scientifically distinguishable.

Where possible, isolate these two factors while preserving a common underlying optimization framework.

The objective is not to invent one arbitrary algorithm per configuration. Instead, investigate the strongest mathematically and experimentally justified rules that can be discovered within each configuration.

Each configuration should be treated as a research problem in its own right.

---

# PHASE 1 — Literature, Mathematical Research, Statistical Analysis, and Formal Verification

This is the foundational and most intellectually intensive phase.

**Do not proceed to large-scale model training before establishing a substantial theoretical, statistical, and experimental foundation.**

## 1.1 Exhaustive literature investigation

Begin with a broad-first literature search across all potentially relevant mathematical, statistical, numerical optimization, machine learning, and computational research fields.

Search beyond terminology such as "Hessian tomography." Different research communities may have solved equivalent or closely related problems using entirely different language.

Investigate all relevant prior work involving directional measurements, matrix information acquisition, curvature approximation, information-efficient optimization, adaptive measurement allocation, adaptive directional selection, and related concepts.

For every important paper, determine:

- The precise problem addressed.
- The underlying mathematical formulation.
- What information is measured and how it is used.
- Whether the number of measurements is fixed or adaptive.
- Whether their directions are fixed or adaptive.
- The objective used to select measurements.
- Theoretical guarantees and assumptions.
- Computational and memory complexity.
- Experimental evidence and limitations.
- Whether the approach can be applied or adapted to our research.
- Its exact overlap with our proposed contribution.

Read and compare the primary papers rather than relying only on abstracts or secondary descriptions.

Identify existing algorithms and theoretical results that could provide inspiration, reusable ideas, applicable benchmarks, or counterexamples to our hypotheses.

Create a rigorous prior-art and novelty assessment.

**Never claim novelty because an exact phrase or paper title was not found.** Seek mathematical and algorithmic equivalence.

Separate established results, direct adaptations, potentially original ideas, and genuinely unresolved questions.

Maintain traceable references and links to primary sources.

Continue revisiting the literature whenever subsequent discoveries reveal new connections.

## 1.2 Deep mathematical investigation

Create a folder named `analysis` for the research investigations.

Perform extremely deep mathematical analysis of the central problem.

Investigate the fundamental limits and achievable performance of extracting optimization-relevant information from directional Hessian measurements.

Study the number of vectors, their arrangement, their interaction, and the consequences of making either or both adaptive.

Research theoretical relationships involving dimensionality, curvature structure, effective information content, measurement quality, optimization accuracy, computational complexity, and uncertainty.

Determine which mathematical objectives meaningfully capture the value of an additional measurement.

Explore competing formulations rather than assuming that one measure of information is universally appropriate.

Derive candidate rules for every experimental configuration.

Analyze both idealized and realistic conditions, including noisy observations, nonconvex loss landscapes, indefinite curvature, time-varying Hessians, and high-dimensional parameter spaces.

Investigate best-case, average-case, worst-case, and practically relevant regimes.

Seek rigorous bounds, optimality conditions, approximation guarantees, and conditions under which one configuration should outperform another.

Identify where the problem is fundamentally impossible, underdetermined, or computationally inefficient.

Do not hide inconvenient mathematical results.

## 1.3 Statistical and Monte Carlo investigations

Support the mathematical reasoning with substantial computational and statistical analysis.

Use Monte Carlo investigations, synthetic experiments, numerical studies, and other appropriate methods to test the hypotheses and explore regimes in which analytic results are incomplete.

Study sensitivity to problem properties, dimensionality, measurement budgets, curvature geometry, noise, initialization, and adaptive decisions.

Use sufficiently diverse problem families to avoid designing methods around a narrow set of favorable examples.

Estimate uncertainty and variability. Distinguish genuine advantages from random effects, favorable test selection, and numerical artifacts.

Search for counterexamples that invalidate proposed general rules.

Compare theoretical predictions with observed behavior and investigate discrepancies.

All experiments must be reproducible, traceable, and computationally accountable.

## 1.4 Systematic optimization of all four configurations

Conduct a broad-first search for existing, applicable, and compatible methods that can inform the design or benchmarking of all four configurations.

Develop and improve the strongest candidate rules for:

**A — Fixed count, fixed alignment**

Find the strongest defensible nonadaptive measurement strategy. Do not automatically assume arbitrary random directions constitute the best fixed baseline.

**B — Fixed count, adaptive alignment**

Investigate how a constant measurement budget can be allocated across changing, information-dependent directions.

**C — Adaptive count, fixed alignment**

Investigate when and how the measurement budget should change while the directional selection remains nonadaptive under the project's formal definition.

**D — Adaptive count, adaptive alignment**

Investigate joint decision rules that adapt both measurement quantity and direction to the state of optimization and previously acquired information.

For each configuration, investigate multiple candidate rules, compare them, retain evidence of unsuccessful approaches, and iteratively improve the strongest candidates.

The investigation must be fair. Do not deliberately weaken any configuration to make the fully adaptive method appear superior.

Compare configurations at relevant matched measurement, computational, and optimization budgets.

Determine whether improvements arise from vector quantity, alignment, their interaction, or simply additional computation.

Use ablations and controlled comparisons to establish causality wherever possible.

**The fully adaptive configuration is a hypothesis, not the predetermined winner.**

## 1.5 Formal verification

Create a folder named `formal`.

Use Lean to formally verify suitable mathematical foundations and significant theoretical results developed during the investigation.

Formalize definitions, assumptions, lemmas, propositions, and important theorems where feasible.

Ensure proofs establish the exact stated results under explicit assumptions.

Do not confuse numerical evidence with mathematical proof.

Do not substitute informal reasoning, unverified external claims, or unproven axioms for completed formal verification.

Avoid proof placeholders in any theorem represented as verified.

When a result cannot be formally established, label it honestly as an open conjecture, conditional result, informal argument, or empirically supported hypothesis.

Use failed proof attempts to identify missing assumptions, invalid claims, and opportunities to strengthen the theory.

Formal verification must inform the research, not merely decorate it.

## 1.6 Living theory document

Maintain a concise file named `theory.md`.

This is **not a research paper**. It is a compact, evolving description of the current scientific understanding.

Include the essential problem definitions, mathematical formulations, important equations, strongest candidate rules, theoretical insights, theorems, assumptions, proof status, central experimental findings, limitations, and unresolved questions.

Keep it readable, precise, and relatively short.

Store detailed derivations, simulation results, investigations, and supporting evidence in `analysis`. Keep Lean artifacts in `formal`.

Whenever mathematical findings, experimental conclusions, algorithms, or formal proofs change, update the relevant research artifacts and `theory.md` so they remain consistent.

Do not allow outdated conjectures or superseded experimental claims to remain presented as established conclusions.

## 1.7 Phase 1 iteration

Phase 1 is iterative, not a one-pass checklist.

Continuously cycle through literature investigation, theory development, mathematical analysis, computational experiments, formal verification, and refinement of the four configurations.

When one idea fails, investigate why.

When a candidate improves, establish what caused the improvement.

When another field suggests a stronger strategy, evaluate it.

Continue seeking stronger candidates until further exploration shows diminishing credible improvement relative to the research budget.

Maintain an explicit record of decisions, rejected hypotheses, open problems, and remaining uncertainties.

Phase 1 is ready to conclude when the four configurations have well-defined, defensible candidate methods; meaningful theoretical and statistical support; verified results where feasible; and enough controlled evidence to justify subsequent language-model experiments.

---

# PHASE 2 — Language Model Training Infrastructure

Prepare the experimental training infrastructure for four approximately **60-million-parameter language models**.

Dataset:

https://huggingface.co/datasets/codelion/fineweb-edu-1B

Use approximately **1 billion training tokens per model** from this dataset.

The four model runs correspond to configurations A, B, C, and D.

Each configuration must train its own independently initialized model instance, using comparable model architectures and controlled training conditions.

Design the architecture, tokenizer handling, preprocessing, sequence construction, training schedule, and evaluation protocol based on sound research and reproducibility considerations.

Keep the model and data conditions as equivalent as possible across the four optimizers.

Prepare reliable training, validation, checkpointing, logging, recovery, and monitoring infrastructure. Profile representative workloads on the available hardware, record model and optimizer costs, and validate numerical correctness and performance before launching the full comparisons.

Capture all optimizer-specific computational overhead.

Ensure every configuration is fully compatible with the training system.

Select implementation details and tools independently, guided by the experimental requirements and available hardware.

**Critical restriction: Phase 2 is preparation only. Do not start any of the four full 1B-token training runs during Phase 2.**

Small correctness checks and infrastructure tests are acceptable.

The full training runs must not begin until the training infrastructure has passed its correctness, recovery, and performance checks.

---

# PHASE 3 — Initial 60M Language Model Comparison

Run the first substantial language-model experiment.

Train four approximately 60M-parameter language models, one for each configuration:

- A: Fixed count, fixed alignment.
- B: Fixed count, adaptive alignment.
- C: Adaptive count, fixed alignment.
- D: Adaptive count, adaptive alignment.

Each model receives approximately **1 billion FineWeb-Edu training tokens**.

Use **one seed per configuration** for this exploratory comparison.

Use the validated training infrastructure developed in Phase 2. Prefer standard XLA execution; retain specialized kernels only when measured end-to-end speed improves and numerical correctness is preserved.

Ensure comparable architecture, training data, token budgets, evaluation conditions, and experimental controls.

Collect comprehensive measurements, including:

- Training and validation loss.
- Perplexity.
- Wall-clock training time.
- Throughput and time to reach specified loss levels.
- Computational consumption and optimizer overhead.
- Memory utilization.
- Optimization stability.
- Curvature measurement counts and costs.
- Behavior of directional selection and adaptation.
- Any other metrics necessary to understand the observed differences.

Analyze both model quality and computational efficiency.

A configuration should not be considered superior merely because it achieves lower loss while consuming substantially more computation.

Perform detailed ablations and analyze the effect of count adaptation, alignment adaptation, and their interaction.

Use the results to select **one winning tomography optimizer** for the next phase.

Define and justify the selection criterion before examining the final Phase 3 results.

Recognize that one seed provides limited statistical evidence. Treat Phase 3 as exploratory screening, not definitive proof of superiority.

Preserve all results, including unfavorable findings.

Update `analysis`, `formal`, and `theory.md` wherever the experiments change our understanding.

---

# PHASE 4 — 600M Hyperparameter Sweeps and Selection

This phase is dedicated to rigorous hyperparameter tuning.

The competitors are:

1. The winning adaptive Hessian tomography optimizer from Phase 3.
2. AdamW.
3. Muon.
4. SOAP.
5. Sophia-G.

Use approximately **600-million-parameter language models**.

For the hyperparameter sweep experiments, use:

- **2,500 training steps per trial.**
- **3 seeds per evaluated hyperparameter setting.**
- Controlled, comparable architectures and training conditions.

Develop and execute systematic hyperparameter searches for each optimizer.

Independently determine appropriate search spaces, tuning strategies, parameter interactions, and relevant stability constraints from existing research and preliminary investigations.

Ensure every optimizer receives a serious and scientifically fair tuning effort.

Do not compare a heavily tuned proposed method against poorly tuned baselines.

Account for differences in tuning cost, per-step execution time, and optimizer-specific computational demands.

Distinguish hyperparameters that transfer reliably across model scales from those requiring caution.

Use independent validation evidence for tuning decisions. Avoid test-set leakage, cherry-picking, and post-hoc modification of selection criteria.

Analyze performance variability across all three seeds.

Investigate interactions between hyperparameters, computational overhead, stability, and convergence.

Record the tuning process comprehensively, including unsuccessful trials and the resources consumed.

At the end of Phase 4, select and freeze the strongest defensible hyperparameter configurations for the five optimizers.

These configurations become the basis of Phase 5.

The 600M sweep is intentionally a different scale from the final 125M experiment. Explicitly investigate whether tuning decisions transfer reliably across those scales; do not assume they do.

Do not use Phase 5 outcomes to retroactively select Phase 4 hyperparameters.

---

# PHASE 5 — Final 125M Language Model Study

Conduct the final controlled comparison between the winning tomography optimizer and the four established competitors.

## Experimental conditions

**Model size:** Approximately 125 million parameters.

**Training budget:** 2.5 billion FineWeb-Edu training tokens per run.

**Seeds:** 3 independent seeds per optimizer.

**Competitors:**

1. Winning Hessian tomography optimizer.
2. AdamW.
3. Muon.
4. SOAP.
5. Sophia-G.

This produces **15 main training runs**.

Use the tuned configurations established in Phase 4, subject only to clearly documented, predeclared scale-transfer rules.

Train all models under matched data, architecture, evaluation, and relevant training conditions.

Use appropriately controlled seeds, with corresponding initialization and data-order conditions matched across optimizers where feasible.

Evaluate not only final performance but also the actual cost of reaching that performance.

## Scientific evaluation

Investigate:

- Final training and validation performance.
- Perplexity.
- Token efficiency.
- Wall-clock efficiency.
- Compute-normalized optimization progress.
- Training throughput.
- Memory requirements.
- Stability and failure rates.
- Variability and uncertainty across seeds.
- Optimizer initialization and state costs.
- Curvature-measurement overhead.
- Practical scaling and implementation complexity.
- Generalization and robustness where justified.

Compare performance under equal-token and equal-computational-resource perspectives.

Report all relevant uncertainty and avoid treating small differences as conclusive.

Assess whether the proposed optimizer delivers a meaningful practical advantage and under what conditions.

A successful outcome need not mean winning every metric.

A result showing superior final loss but inferior computational efficiency is scientifically different from a result showing superior loss at matched compute.

Additionally, retain the relevant results from earlier curvature-method comparisons so that any advantage attributed specifically to adaptive tomography is supported by mechanism-level evidence.

Do not confuse superiority over first-order optimizers with proof that adaptive measurement selection is better than established curvature-acquisition methods.

---

# Global Scientific Standards

These standards apply to every phase.

## 1. Research integrity

Treat every claim as something to establish, not something to defend.

Actively seek falsifying evidence and competing explanations.

Do not manufacture results, references, statistical significance, experimental success, or formal proofs.

Distinguish hypotheses, derivations, verified theorems, numerical observations, and empirical conclusions.

If a proposed technique underperforms, report it honestly.

If existing methods already solve an equivalent problem, acknowledge that overlap.

## 2. Fair experimental comparisons

Predefine important comparison criteria before final evaluations.

Use meaningful control groups and ablations.

Avoid confounding algorithmic improvements with implementation quality, additional compute, unequal tuning, or differing model conditions.

Where feasible, make both equal-token and equal-resource comparisons.

Investigate the marginal contribution of every adaptive component.

## 3. Reproducibility

Make the entire investigation reproducible.

Preserve sufficient information about datasets, preprocessing, model configurations, experiment settings, seeds, measured costs, software environments, and results.

Record failed experiments as well as successful ones.

Support numerical claims with actual experimental evidence.

Ensure the final conclusions can be traced to the research artifacts that justify them.

## 4. Continuous theory–experiment synchronization

The project is expected to evolve substantially.

When experiments contradict the theory, investigate and revise the theory.

When mathematics predicts a better method, test it.

When formal verification reveals an invalid assumption, correct both the mathematical analysis and experimental interpretation.

When the optimal candidate changes, update the corresponding documentation and dependent experiments.

Keep `theory.md` short, accurate, and current throughout.

## 5. Autonomous decision-making

You are expected to research and make technical decisions independently.

Do not request instructions for ordinary implementation choices, tool selection, research methodologies, numerical analysis, or experimental procedures.

Use the available computational resources responsibly.

For major resource commitments, establish the expected compute and storage costs, check available budgets and permissions, and avoid uncontrolled experimentation.

Do not interpret "iterate until best" as unlimited trial-and-error. Use evidence-based stopping criteria, diminishing returns, and resource-aware research planning.

If practical constraints prevent a phase from being completed, document the limitations, complete what can genuinely be established, and do not represent unexecuted experiments as completed.

## 6. Continuous critical review

At each major milestone, assess:

- What have we established?
- What is still uncertain?
- Which previous assumptions have failed?
- Which competing approaches remain plausible?
- What evidence supports the current best configuration?
- What would disprove our current conclusion?
- What is the highest-value next investigation?

Prefer investigations that resolve important uncertainty over investigations that merely generate more results.

---

# Completion Criteria

The project is complete only when the research has produced:

1. A well-supported understanding of the prior literature and the defensible novelty of the proposed approach.
2. Rigorous mathematical and statistical investigations into measurement count, alignment, and adaptation.
3. Strong, systematically developed candidate rules for all four configurations.
4. Formal Lean verification of important theoretical results where feasible, with clearly identified unproved claims.
5. A concise and consistently updated `theory.md`.
6. Reproducible Phase 1 analysis and simulation evidence.
7. Tested and optimized language-model training infrastructure.
8. Completed Phase 3 exploratory comparisons across four configurations.
9. Completed Phase 4 hyperparameter searches across the five competitors.
10. Completed Phase 5 final comparisons, with reliable evaluation across three seeds.
11. An evidence-based understanding of when adaptive Hessian tomography helps, when it fails, and whether it offers an advantage over existing optimization methods.

Do not prioritize producing a positive result over producing a correct result.

# Ultimate Research Objective

Determine whether intelligently controlling **how many directional curvature measurements are acquired and how those measurements are aligned** can yield an optimization algorithm that captures enough useful Hessian information to outperform existing alternatives under realistic computational constraints.

The central scientific contribution, if supported, should be a principled and experimentally validated answer to:

**Given the current optimization state and a limited computational budget, how many Hessian measurement vectors should we use, where should we position them, and how should these decisions change as new information becomes available?**

Discover the answer through theory, statistical evidence, formal verification, and rigorous experiments.

**The objective is not to prove that our idea works. The objective is to discover whether it works, why it works, where it works, and whether it genuinely improves upon what already exists.**
