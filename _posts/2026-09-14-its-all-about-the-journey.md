---
title: "Consensus doesn't induce stability"
date: 2026-09-14 12:00:00 +0000
categories: [Research, Ideas]
tags: [decentralised_systems, self_organisation, incentive_design]
math: true
description: Sheaf Diffusion, Coordination Graphs, and Local Certificates for Decentralised Learning
---

> **Working thesis.** Literature research shows that sheaf geometry determines which disagreement modes a decentralised coordination mechanism can observe and damp. Local objectives and game dynamics determine whether the remaining modes are stable or desirable. Decentralised optimisation methods like ADMM or VDN adds primal-dual feedback for enforcing compatibility. Hodge decomposition and cycle transport then provide complementary (howevever, fairly deemed incomplete) views of local-to-global inconsistency.

## Naïve introduction

Decentralised learning has a naïve baseseline implemenation  through a scalar consensus matrix that mixes every coordinate of every agent state in the same way. There is nothing wrong in doing in that. Moreover, it is appropriate when the desired solution is full agreement, but it is too restrictive when agents should coordinate only through selected public interfaces while retaining private or heterogeneous behaviour (take as an example partial observability or role models in multi-agent systems). [Coordination graphs](https://arxiv.org/abs/2606.02337) offer quite different but natural decomposition, factoring global value function into regional terms and reconcile overlapping actions through message passing. Recent constrained coordination-graph methods combine this factorisation with Lagrangian duality, but cyclic factor graphs still require heuristic damping and overlapping regions can produce structural incompatibilities.

However, most of the approaches develop a framework around vanilla message passing, not taking into account system's topology, for example, sheaves. Sheaf restriction maps generalise identity overlap constraints by specifying which transformations of local states must agree. The associated sheaf Laplacian generates diffusion—like dynamics that damp inconsistent modes while preserving heterogeneous global sections. Adding local strategic objectives produces a sheaf reaction–diffusion system; adding dual variables produces a primal–dual feedback system closer to the method called [Sheaf-ADMM](https://pub.sakana.ai/sheaf-admm/). We connect its undamped modes to Hodge theory and cycle consistency to monodromy or, for general Jacobian blocks, loop-gain operators. A preliminary triangle example shows both the promise and limitation of cycle diagnostics: they can detect interaction-induced tension but may miss stabilising self-dynamics. 

We'll conclude with a staged research program separating expressivity, optimisation, topology, stability, stochasticity, and strategic learning.

---

## Why agreement is not enough

### The empirical starting point

In one my past works, I [argued](https://alexunderch.github.io/posts/better-together/) that complexification of the Streaming RL problem makes its solution more flexible, but still introduces gradient variance-consensus tradeoff. Notably, we concluded it as an **empirical observation**, not as a universal impossibility theorem about decentralised SGD.


Phrasing it differently, the motivating observation was that changing a conventional consensus matrix did not independently control:

1. topological agreement between agents;
2. the covariance or variance of stochastic gradients;
3. preservation of agent-specific behaviour;
4. the stability of the resulting learning dynamics.

You can read my observation in the blog, link above.

### A concerning theoretical claim

Let each of $n$ agents hold a state $x_i\in\mathbb R^d$, and stack the states into $x\in\mathbb R^{nd}$. A coordinate-blind mixing operator has the form

$$
\mathcal W = W\otimes I_d.
$$

Let $P$ project onto a public subspace and $Q=I_d-P$ onto a private subspace. Therefore, a desired selective operator is

$$
\mathcal T
=
W_P\otimes P
+
W_Q\otimes Q.
$$

For example, choosing $W_Q=I_n$ mixes public variables while preserving private variables:

$$
\mathcal T
=
W_P\otimes P
+
I_n\otimes Q.
$$

### Proposition 1 — Expressivity of coordinate-blind consensus

If $P$ is nontrivial and $W_P\neq W_Q$, there is no matrix $W$ such that

$$
W\otimes I_d
=
W_P\otimes P+W_Q\otimes Q.
$$

<details>

<summary>Proof sketch</summary>



On $\mathbb R^n\otimes\operatorname{im}(P)$, equality requires $W=W_P$. On $\mathbb R^n\otimes\operatorname{im}(Q)$, it requires $W=W_Q$. Therefore, the equality can hold only when $W_P=W_Q$.

</details>


This result concerns coordinate-blind mixing of the form $W\otimes I_d$. It does **not** rule out a lot of subsequent research that really works:

- matrix-weighted gossip;
- gradient tracking;
- exact-diffusion methods;
- personalized decentralised optimization;
- coordinate-adaptive or time-varying communication;
- and so on...

Its purpose is to motivate a richer communication operator, not to claim that every method called D-SGD must fail.

### Conditional covariance consequence

Let gradient noise split into public and private components,

$$
\xi=\xi_P+\xi_Q,
\qquad
\xi_P=(I_n\otimes P)\xi,
\qquad
\xi_Q=(I_n\otimes Q)\xi.
$$

Selective mixing acts as

$$
\mathcal T\xi
=
(W_P\otimes P)\xi_P
+
(W_Q\otimes Q)\xi_Q.
$$

Under an appropriate covariance decomposition, $W_P$ and $W_Q$ independently determine the node-space filtering of the two noise components. Coordinate-blind mixing applies the same $W$ to both.

> **Claim to prove!** Under explicit assumptions on update order, noise covariance, and objective geometry, selective mixing permits variance attenuation in public directions without imposing the same attenuation or consensus dynamics on private directions.


## Coordination graphs: decentralization without parameter averaging

### Regional factorisation

A coordination graph replaces a monolithic value function with regional factors:

$$
Q(s,a)\approx\sum_{R\in\mathcal C_R}Q_R(s^R,a^R).
$$

[Recent](https://arxiv.org/abs/2606.02337) constrained coordination-graph MARL augments each regional objective with constraint values:

$$
Q_{\mathrm{aug},R}
=
Q_{\mathrm{prim},R}
+
\lambda^\top Q_{\mathrm{cost},R}.
$$

Joint actions are selected through **Max-Sum** message passing over the induced factor graph.

### Overlap as a gluing problem

Suppose regions $R$ and $S$ both contain agent $i$. Introduce regional copies $a_i^R, \qquad a_i^S.$

Ordinary coordination requires identity consistency $a_i^R=a_i^S.$

When separately preferred regional actions disagree, the regional solutions do not immediately glue into a single joint action. This produces two distinguishable difficulties:

1. **structural error:** the regional factorization is not exact;
2. **solver error:** loopy message passing does not reach the exact optimum of the factorised problem.

### Why cycles matter

Max-Sum is exact under standard assumptions on trees, but loopy graphs may oscillate or fail to converge. Practical systems therefore introduce message damping:

$$
m^{k+1}
=
dm^k+(1-d)T(m^k).
$$

Near a fixed point $m^\star$,

$$
\Delta m^{k+1}
\approx
\left[dI+(1-d)J_T(m^\star)\right]\Delta m^k.
$$

Local convergence requires

$$
\rho\left(dI+(1-d)J_T(m^\star)\right)<1.
$$

This gives a concrete, conventional message-passing system on which cycle diagnostics and local stability bounds can be tested.

### Summary

The coordination-graph literature provides a second motivation alongside D-SGD:

- D-SGD reveals an expressivity limitation of coordinate-blind state mixing;
- coordination graphs avoid parameter averaging, but expose overlap compatibility and loopy-message stability problems.

The proposed sheaf/ADMM framework should be presented as a generalisation of overlap consistency, not merely as another averaging method.

## Sheaves generalise what overlapping regions must agree on

### Restriction maps as public interfaces

For an edge or overlap $e=(i,j)$, assign maps

$$
F_{i\to e}:V_i\to V_e,
\qquad
F_{j\to e}:V_j\to V_e.
$$

Compatibility means

$$
F_{i\to e}x_i
=
F_{j\to e}x_j,
$$

rather than necessarily $x_i=x_j$.

The sheaf coboundary is

$$
(\delta_{\mathcal F}x)_e
=
F_{i\to e}x_i-F_{j\to e}x_j.
$$

The compatibility space is

$$
H^0(G;\mathcal F)
=
\ker\delta_{\mathcal F}.
$$

### Private and public directions

A direction at vertex $i$ is locally invisible to all incident comparisons if it lies in

$$
\bigcap_{e\ni i}\ker F_{i\to e}.
$$

However, useful heterogeneous behavior need not be completely invisible. It may instead form a nonconstant global section in $\ker\delta_{\mathcal F}$.

This distinction should be maintained:

- **locally private directions:** invisible to incident restrictions;
- **globally consistent heterogeneous directions:** visible but mutually compatible;
- **inconsistent directions:** penalised by the sheaf geometry.

### Coordination-graph specialisation

Identity overlap constraints are recovered when $F_{R\to O}$ simply selects the action coordinates shared by regions $R$ and $S$:

$$
F_{R\to O}x_R=F_{S\to O}x_S.
$$

A nontrivial sheaf allows projected, transformed, or compressed compatibility. This is the main structural extension beyond ordinary coordination graphs.

## Sheaf diffusion: the geometry of selective damping

The sheaf Laplacian is

$$
L_{\mathcal F}
=
\delta_{\mathcal F}^*\delta_{\mathcal F}.
$$

Pure sheaf diffusion is

$$
\dot x=-L_{\mathcal F}x,
$$

with solution

$$
x(t)=e^{-tL_{\mathcal F}}x(0).
$$

If

$$
L_{\mathcal F}v_k=\lambda_kv_k,
$$

then each mode evolves as

$$
e^{-\lambda_k t}v_k.
$$

Therefore:

- $\lambda_k>0$: the mode is damped;
- $\lambda_k\approx0$: the mode reconciles slowly;
- $\lambda_k=0$: the mode survives as a global section.

### Elementary public/private example

If every edge compares the same public projection $P,$ then

$$
L_{\mathcal F}=L_G\otimes P.
$$

Consequently,

$$
e^{-tL_{\mathcal F}}
=
e^{-tL_G}\otimes P
+
I_n\otimes Q.
$$

This is the selective operator that coordinate-blind consensus cannot represent when public and private directions require different dynamics.

### The design question

The goal is not to maximise the sheaf spectral gap without qualification. It is to choose the kernel and positive spectrum so that:

$$
\text{desirable heterogeneous modes}
\subseteq
\ker L_{\mathcal F},
$$

while

$$
\text{undesirable inconsistent modes}
\subseteq
(\ker L_{\mathcal F})^\perp.
$$

---

## From diffusion to constrained optimization and ADMM

### Penalty formulation

Consider

$$
\min_x f(x)
\quad\text{subject to}\quad
\delta_{\mathcal F}x=0.
$$

A quadratic-penalty objective is

$$
f(x)
+
\frac{\rho}{2}\|\delta_{\mathcal F}x\|^2.
$$

Its gradient flow contains sheaf diffusion:

$$
\dot x
=
-\nabla f(x)-\rho L_{\mathcal F}x.
$$

At finite $\rho$, a quadratic penalty need not produce exact feasibility. Taking $\rho$ very large may worsen conditioning.

### Augmented Lagrangian

Introduce a dual variable $u$:

$$
\mathcal L_\rho(x,u)
=
f(x)
+
\langle u,\delta_{\mathcal F}x\rangle
+
\frac{\rho}{2}\|\delta_{\mathcal F}x\|^2.
$$

A continuous primal–dual analogue is

$$
\begin{aligned}
\dot x
&=
-\nabla f(x)
-\delta_{\mathcal F}^*u
-\rho L_{\mathcal F}x,\\
\dot u
&=
\delta_{\mathcal F}x.
\end{aligned}
$$

Interpretation:

- sheaf diffusion provides proportional damping of compatibility errors;
- the dual variable provides integral feedback;
- the combined system is not a pure heat equation;
- primal–dual coupling can produce oscillatory transients.

### Two distinct dual layers in constrained MARL

A constrained coordination problem may contain both:

1. a CMDP multiplier $\lambda$ pricing safety, resource, or welfare constraints;
2. a sheaf/overlap dual $u$ enforcing compatibility between regional variables.

A schematic objective is

$$
\max_{\{x_R\}}
\sum_R
\left[
Q_{0,R}(x_R)
+
\lambda^\top Q_{c,R}(x_R)
\right]
\quad
\text{subject to}
\quad
\delta_{\mathcal F}x=0.
$$

The two dual variables answer different questions:

$$
\lambda:
\text{Are task-level constraints satisfied?}
$$
and
$$
u:
\text{Are local or regional representations compatible?}
$$

> They should not be interpreted as interchangeable decision-making knobs!



## Strategic dynamics as sheaf reaction–diffusion

Let the agents have losses $\ell_i(x_i,x_{-i})$ and joint pseudo-gradient

$$
\xi(x)
=
\begin{bmatrix}
\nabla_{x_1}\ell_1(x)\\
\vdots\\
\nabla_{x_n}\ell_n(x)
\end{bmatrix}.
$$

Uncoordinated gradient play is

$$
\dot x=-\xi(x).
$$

Add sheaf coordination:

$$
\boxed{
\dot x
=
-\xi(x)-\rho L_{\mathcal F}x.
}
$$

This is a strategic reaction–diffusion system:

- $-\xi(x)$ is strategic reaction;
- $-\rho L_{\mathcal F}x$ is topological diffusion.

### Linearised stability

At a sheaf-consistent stationary point $x^\star$, let

$$
J=D\xi(x^\star).
$$

Then

$$
\dot y
\approx
-(J+\rho L_{\mathcal F})y.
$$

Decompose

$$
J=S+A,
$$

where

$$
S=\frac{J+J^\top}{2},
\qquad
A=\frac{J-J^\top}{2}.
$$

For $V(y)=\frac12\|y\|^2$,

$$
\dot V
=
-y^\top(S+\rho L_{\mathcal F})y.
$$

Therefore,

$$
S+\rho L_{\mathcal F}\succ0
$$

*is a sufficient local contraction condition*.

### The undamped-mode limitation

If $h\in\ker L_{\mathcal F}$, then

$$
L_{\mathcal F}h=0.
$$

Sheaf diffusion cannot stabilise game instabilities lying completely in the global-section space. Such modes require stabilization by:

- local curvature or regularisation;
- incentive design;
- an alternative game-learning dynamic;
- additional restrictions;
- centralized or global feedback.

### Primal–dual strategic dynamics

Adding the dual gives the linearised block system

$$
\begin{bmatrix}
\dot y\\
\dot\nu
\end{bmatrix}
=
\begin{bmatrix}
-(J+\rho L_{\mathcal F}) & -\delta_{\mathcal F}^*\\
\delta_{\mathcal F} & 0
\end{bmatrix}
\begin{bmatrix}
y\\
\nu
\end{bmatrix}.
$$

This combined operator—not the sheaf Laplacian alone—determines stability.

## Hodge decomposition and cycle transport

### Additive Hodge view

For prescribed edge data $b\in C^1$, a Hodge decomposition has the schematic form

$$
b
=
\delta_{\mathcal F}\phi
+
h
+
(\delta_{\mathcal F}^{1})^*\psi.
$$

Interpretation:

- $\delta_{\mathcal F}\phi$: globally explainable by vertex data;
- $h$: harmonic/global cyclic component;
- $(\delta_{\mathcal F}^{1})^*\psi$: local curl around filled cells.

> On a graph without $2$-cells, the coexact term is absent.

### Important distinction

If an ADMM residual is defined as

$$
r=\delta_{\mathcal F}x,
$$

then it is exact by construction. A nontrivial cohomological obstruction appears when solving a prescribed-data problem

$$
\delta_{\mathcal F}x=b
$$

and asking which part of $b$ cannot be represented as a coboundary.

### Multiplicative monodromy view

For invertible transports $\rho_e$, define monodromy around a cycle $C$ by

$$
M_C
=
\rho_{e_k}\cdots\rho_{e_1}.
$$

A global parallel state must satisfy

$$
M_Ch=h
$$

for every cycle. Thus the global-section space is related to the common fixed space

$$
H^0(G;\rho)
\cong
\bigcap_C\ker(M_C-I).
$$

Monodromy is:

- multiplicative;
- cycle-specific;
- sensitive to ordering for matrix-valued transports.

Hodge decomposition is:

- additive;
- system-wide;
- dependent on the chosen inner products and complex.

### Abelian correspondence

For scalar phase transports

$$
\rho_e=e^{ia_e},
$$

cycle monodromy is

$$
M_C
=
\exp\left(i\sum_{e\in C}a_e\right).
$$

On a one-dimensional graph, harmonic circulation can be interpreted as the additive logarithm of monodromy modulo periodicity.

### Noncommutative gap

For matrix transports $\rho_e=e^{A_e}$,

$$
\log M_C
\neq
\sum_{e\in C}A_e
$$

in general. The difference contains commutator terms such as

$$
[A_e,A_f].
$$

This produces a concrete research question:

>**Question to prove!** Under what commutativity, small-curvature, or near-identity assumptions does additive Hodge information approximate multiplicative monodromy well enough to predict dynamical behaviour?


<!-- ## A worked triangle: what cycle information sees and misses

### Setup

[Insert the precise three-agent Hawk–Dove or quadratic-game definition.]

Let the linearized game Jacobian be

$$
J(c)=D(c)+B,
$$

where:

- $D(c)$ contains self-dynamics and local regularization;
- $B$ contains cross-agent interactions.

## 8.2 Initial hypothesis

The cycle product of local cross-agent response maps might indicate whether the equilibrium is stable.

For a triangle, define schematically

$$
M_{	riangle}
=
J_{1\leftarrow3}
J_{3\leftarrow2}
J_{2\leftarrow1}.
$$

## 8.3 Positive observation

At the unregularized point, the cycle diagnostic signals interaction-induced tension, agreeing with the independently computed unstable eigenstructure.

## 8.4 Counterexample

As $c$ increases, the true stability changes because $D(c)$ changes. The off-diagonal cycle product remains constant because it only sees $B$.

Therefore,

$$
\text{cycle feedback alone}
\neq
\text{full dynamical stability}.
$$ -->

### Composite local diagnostics

A Gershgorin-type certificate combines diagonal self-dynamics with off-diagonal interaction magnitude. It may be sound but conservative.

Compare:

1. cycle loop gain;
2. ordinary Gershgorin bounds;
3. weighted Gershgorin bounds;
4. Brauer-type bounds;
5. local approximations to sheaf spectral modes;
6. exact spectral abscissa as ground truth.


> Cycle transport diagnoses cyclic coupling. Local diagonal terms diagnose self-reaction. The sheaf Laplacian diagnoses available diffusive damping. Full stability depends on all three and on their interaction.


## Certification table

| Property | Mathematical object | Candidate diagnostic |
|---|---|---|
| Compatibility feasibility | Sheaf constraints | $\|\delta_{\mathcal F}x\|$ |
| Integrability of prescribed edge data | Cohomology/Hodge residual | $\|P_{H^1}b\|$ |
| Diffusive convergence | Sheaf Laplacian | spectral gap and kernel |
| Strategic local stability | Combined Jacobian | spectral abscissa or Lyapunov test |
| Cyclic feedback | Cycle transports | monodromy or loop gain |
| Message-passing stability | Message-update Jacobian | $\rho(J_T)$ |
| Incentive to deviate | Utilities/game dynamic | regret or equilibrium gap |
| Constraint satisfaction | CMDP or resource constraints | primal violation and $\lambda$ |
| Social efficiency | Welfare | welfare gap or Price of Anarchy |
| Stochastic coordination benefit | Gradient statistics | projected covariance |
| Communication sufficiency | Information structure | reconstruction/certification error |

>These are distinct properties. They can be represented and compared within a shared local-to-global architecture—not that they are identical.


The coordination-graph literature suggests separating sources of value error:

$$
\text{value error}
\lesssim
\text{factorization error}
+
\text{coordination-solver error}
+
\text{sampling error}
+
\text{function-approximation error}.
$$

The proposed framework adds separate measurements for:

- compatibility feasibility;
- dual convergence;
- topological/Hodge obstruction;
- dynamical stability margin;
- stochastic covariance;
- welfare or equilibrium quality.



## Experimental program

### Phase 1: Linear expressivity

Use a two-agent, two-coordinate public/private example.

Compare:

- coordinate-blind mixing $W\otimes I$;
- matrix-weighted selective mixing;
- sheaf diffusion $L_G\otimes P$.

Supposed to verify Proposition 1 and the public/private covariance consequence.

### Phase 2: Fixed coordination problems

Freeze regional Q-tables or continuous regional utilities. Compare:

1. exact centralized joint optimization;
2. undamped Max-Sum;
3. damped Max-Sum;
4. consensus ADMM on relaxed regional variables;
5. Sheaf-ADMM with nontrivial restrictions.

Measure:

- objective gap;
- overlap residual;
- convergence rate;
- oscillation;
- cycle diagnostic values;
- Hodge residuals;
- certificate accuracy.

For discrete actions, document the continuous relaxation and any rounding/integrality gap.

### Phase 3: Quadratic network games

Use

$$
\ell_i(x)
=
\frac12x_i^\top Q_ix_i
+
\sum_{j\in\mathcal N(i)}x_i^\top A_{ij}x_j
-b_i^\top x_i.
$$

Generate:

- potential/symmetric games;
- Hamiltonian or zero-sum games;
- mixed games with tunable symmetric and antisymmetric components.

Compare:

$$
\dot x=-\xi(x),
$$

$$
\dot x=-\xi(x)-\rho L_{\mathcal F}x,
$$

and the primal–dual system.

### Phase 4: Topological complexity

Progress through:

1. paths;
2. one unfilled cycle;
3. filled triangles;
4. overlapping cycles;
5. long-cycle graphs;
6. low-rank or lossy restrictions;
7. time-varying interaction graphs.

Compute:

- $H^0$ and, where appropriate, $H^1$;
- sheaf spectral gaps;
- harmonic projections;
- monodromies or loop gains;
- exact full-system spectra.

### Phase 5: Stochastic and asynchronous optimisation

Add:

- stochastic gradient estimates;
- random edge activation;
- bounded communication delays;
- noisy messages;
- heterogeneous update frequencies.

Measure:

- public/private projected covariance;
- primal and dual residuals;
- stability-prediction accuracy;
- convergence under stragglers;
- retained heterogeneous behavior.

### Phase 6: Strategic learning extensions

Only after the optimization and stability mechanisms are validated, compare:

- gradient play;
- no-external-regret or swap-regret updates;
- variational-inequality methods;
- explicit opponent models;
- one-step opponent shaping;
- DiCE/LoadedDiCE when stochastic higher-order policy gradients are required.

Evaluated using equilibrium gap, stability, constraint satisfaction, welfare, and Price of Anarchy.

<details>

<summary>Optional strategic branches</summary>


### No-regret learning

No-regret methods provide an equilibrium-learning mechanism, not a stability or welfare certificate. A possible result is a perturbed-regret bound of the form

$$
\operatorname{CCEGap}(T)
\lesssim
\frac{\sum_iR_i(T)}{T}
+
\frac1T\sum_{t=1}^T\epsilon_t,
$$

where $\epsilon_t$ measures error in sheaf-mediated belief or payoff reconstruction.

### Opponent modelling and shaping

Opponent modelling asks how another agent adapts. DiCE and LoadedDiCE are estimators for differentiating stochastic objectives, including objectives that differentiate through another learner's update.

A possible question is:

> Does differentiating through several Sheaf-ADMM or message-passing iterations improve welfare and stability, or does it introduce additional variance and strategic manipulation?

### Price of Anarchy

Price of Anarchy evaluates equilibrium efficiency. It is not an update rule and should not be treated as an alternative to no-regret learning or opponent modelling.

Use it as one output metric:

$$
\operatorname{PoA}
=
\frac{\text{optimal welfare}}
{\text{worst equilibrium welfare}}
$$

under a welfare-maximization convention.


</details>


## Research questions

### RQ1 — Selective stabilisation

Can restriction maps be designed so that undesirable unstable modes lie in the damped subspace while useful heterogeneous modes remain in $\ker L_{\mathcal F}$?

### RQ2 — Local certification

Which combinations of diagonal terms, loop gains, and local sheaf spectral information provide useful sufficient conditions for the stability of

$$
J+\rho L_{\mathcal F}
$$

or the corresponding primal–dual block operator?

### RQ3 — Hodge versus monodromy

When do additive harmonic diagnostics approximate multiplicative cycle transport? How does the approximation degrade with noncommutativity?

### RQ4 — Max-Sum versus ADMM

For which continuous or relaxed coordination problems can ADMM provide a more stable or more diagnosable alternative to loopy Max-Sum? Where does Max-Sum remain preferable because of discrete integrality?

### RQ5 — Structural error and gluing obstruction

Under smoothness, strong concavity, or unique-maximizer assumptions, can regional incompatibility or a Hodge residual bound the structural factorization error of a coordination graph?

### RQ6 — Primal–dual mode dynamics

Which Hodge modes of the augmented-Lagrangian system are overdamped, underdamped, oscillatory, or unaffected by compatibility constraints?

### RQ7 — Stochastic and asynchronous robustness

Do deterministic local certificates remain predictive under gradient noise, delayed messages, random edge activation, and time-varying restrictions?

### RQ8 — Mixed intentions

After the cooperative optimization case is understood, what changes when regional objectives belong to different strategic agents rather than a shared team objective?


## Results that appear straightforward but require formal proof

- The coordinate-blind expressivity proposition.
- The public/private covariance corollary under a specified stochastic model.
- Mode-wise stability conditions when $J$ and $L_{\mathcal F}$ share invariant subspaces.
- Exact single-cycle relationships between monodromy and twisted cohomology in the chosen setup.

## Research hypotheses

- Hodge and loop-gain diagnostics can predict practically relevant instability.
- Sheaf-valued overlaps reduce structural incompatibility in factorized coordination.
- ADMM feedback is more stable or diagnosable than damped Max-Sum in an identifiable class of problems.
- Learned restrictions can stabilize public modes while preserving useful private diversity.
- Local certificates remain informative under stochasticity and asynchrony.
- Improved compatibility and stability lead to better equilibrium welfare or PoA.




## (Provisional) conclusion

There was proposed a way of thinking that does not identify consensus, equilibrium, stability, and welfare as the same object. Instead, it separates the layers of a decentralised system:

1. the **communication geometry** specifies what can be compared;
2. the **sheaf Laplacian** damps incompatible modes;
3. the **local objective or game** supplies reaction, rotation, and incentives;
4. **ADMM dual variables** add feedback for enforcing compatibility;
5. **Hodge decomposition** separates explainable and cyclic edge data;
6. **monodromy or loop gain** measures accumulated transport around selected cycles;
7. the **combined operator** determines actual dynamical stability;
8. regret, constraint violation, and PoA evaluate distinct strategic properties.

The central frontier, therefore, can be stated as:

> Can locally available geometric and dynamical quantities provide useful guarantees about global coordination without collapsing the heterogeneous modes that make decentralized learning valuable?

## Some references

1. Seely, J., Cupiał, B., and Jones, L. *Learning Multi-Agent Coordination via Sheaf-ADMM*. 2026.
2. Bodnar, C., Di Giovanni, F., Chamberlain, B., Liò, P., and Bronstein, M. *Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs*. NeurIPS 2022.
3. Hansen, J., and Ghrist, R. *Toward a Spectral Theory of Cellular Sheaves*. Journal of Applied and Computational Topology, 2019.
4. [Authors]. *Coordination Graphs for Constrained Multi-Agent Reinforcement Learning*. arXiv:2606.02337, 2026.
5. Guestrin, C., Koller, D., and Parr, R. *Multiagent Planning with Factored MDPs*. NeurIPS 2001.
6. Balduzzi, D., et al. *The Mechanics of n-Player Differentiable Games*. ICML 2018.
7. Candogan, O., Menache, I., Ozdaglar, A., and Parrilo, P. *Flows and Decompositions of Games: Harmonic and Potential Games*. Mathematics of Operations Research, 2011.
8. Jiang, X., Lim, L.-H., Yao, Y., and Ye, Y. *Statistical Ranking and Combinatorial Hodge Theory*. Mathematical Programming, 2011.
9. Koloskova, A., et al. *A Unified Theory of Decentralized SGD with Changing Topology and Local Updates*. ICML 2020.
10. Yuan, K., Ying, B., Zhao, X., and Sayed, A. *Exact Diffusion for Distributed Optimization and Learning*. 2017–2019.
11. Hong, M., Luo, Z.-Q., and Razaviyayn, M. *Convergence Analysis of ADMM for a Family of Nonconvex Problems*. SIAM Journal on Optimization, 2016.
12. Chen, C., He, B., Ye, Y., and Yuan, X. *The Direct Extension of ADMM for Multi-Block Convex Minimization Problems Is Not Necessarily Convergent*. Mathematical Programming, 2016.
13. Hart, S., and Mas-Colell, A. *A Simple Adaptive Procedure Leading to Correlated Equilibrium*. Econometrica, 2000.
14. Hart, S., and Mas-Colell, A. *Uncoupled Dynamics Do Not Lead to Nash Equilibrium*. American Economic Review, 2003.
15. Wolpert, D., and Tumer, K. *Optimal Payoff Functions for Members of Collectives*. Advances in Complex Systems, 2001.

