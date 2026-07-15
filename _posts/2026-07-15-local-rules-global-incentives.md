---
title: "Self-Organisation Lacks a Global Incentive"
date: 2026-07-15 14:00:00 +0000
categories: [Research, Ideas]
tags: [decentralised_systems, self_organisation, incentive_design]
math: true
description: Local rules might work great but they still have to lead to the global incentive.
---

I have recenly found about the paper that felt for me like a breath of fresh air amidst the the casual multi-agent or self-organisation liteature, [Sheaf-ADMM](https://pub.sakana.ai/sheaf-admm/). The method self-organises a group of agents into a coherent global solution
using nothing but local communication and a shared dual variable. It's a genuinely nice piece of machinery. 

<iframe src="https://pub.sakana.ai/sheaf-admm/" width="100%" height="500px" style="border: 1px solid var(--card-border-color); border-radius: 8px; background: white;"></iframe>


However, as was reading the paper and even talking with its main author (nice lad!), I started feeling something a bit of about the paper's objective: nothing in it asks whether an agent *wants* to be there. In other words, *is there an equilibrium the agent aims to achieve?* Therefore, in this I would like to discuss what might be missing, why the obvious fixes don't close the gap, and what a fix that actually comes with a guarantee looks like.

> The point of this piece is to look at the problem through a different lens.
{: .prompt-info }

##  The problem, re-stated

**The game.** Consider a graphical (polymatrix) game on an undirected graph $G = (V,E)$. Each agent $i$ has a finite action set $A_i$ and plays a mixed strategy $\pi_i$ on the simplex $\Delta(A_i)$. Agent i interacts only with its neighbours $\mathcal{N}(i) = \{j : (i,j) \in E\}$, and its payoff is additively separable
across those interactions:
$$
    U_i(\pi_i, \pi_{\mathcal{N}(i)}) = \sum_{j \in \mathcal{N}(i)}  E_{a_i\sim\pi_i, a_j\sim\pi_j} [ u_i^{(i,j)}(a_i,a_j) ]
$$
where $u_i^{(i,j)}: A_i \times A_j \to \mathbb{R}$ is agent $i^{\text{th}}$ payoff from its pairwise game with neighbor $j$. This is exactly the structure Sheaf-ADMM is built for:
local objectives, local communication, no global payoff tensor.

**What Sheaf-ADMM actually optimises.** Give each agent a state $x_i$ in some
stalk space, and a restriction map $F_{ij}$ sending that state into a shared edge
space. Sheaf-ADMM solves:

$$
\begin{aligned}
    \min    \sum_i f_i(x_i), \\
    \text{subject to }  F_{ij}(x_i) = F_{ji}(x_j) \quad    \text{for every } (i,j) \in E
\end{aligned}
$$

via the augmented Lagrangian

$$
\begin{aligned}
    L_\rho(x,u) = \sum_i f_i(x_i)
                 + \sum_{(i,j)} u_ij^T (F_{ij}(x_i) - F_{ji}(x_j)) \\
                 + (\rho/2) \sum_{(i,j)} || F_{ij}(x_i) - F_{ji}(x_j) ||^2
\end{aligned}
$$

The solver unrolls a standard loop: an $x$-update (minimise $L_\rho$ over each $x_i$, holding the rest fixed) and a dual-ascent $u$-update:

$$
    u_{ij}  \leftarrow  u_{ij} + \rho  (F_{ij}(x_i) - F_{ji}(x_j) )
$$


I saw a problem into the objective: **$f_i$ is described as "a
convex function," and it we don't know much about $U_i$.** Its elements have such a "handcrafted" cooperative representation, which make me question what would change if we tried to solve a tension between the agents who actually compete. [^1] 


## Notions of "good outcome", or equilibrium, in multi-agent systems

Fix a joint distribution $\sigma$ over the full joint action space
$A = A_1 \times \ldots \times A_n$. (Not a product distribution necessarily, that
generality is the whole point of what follows.)

**Nash equilibrium (NE).**  $\sigma$  is required to be a product,
$\sigma = \pi_1 \times \ldots \times \pi_n$, and no agent can improve by unilaterally
changing its own marginal:

$$
    U_i(\pi_i, \pi_{-i})  \geq  U_i(\pi_i', \pi_{-i}) \quad   \text{for every } \pi_{i'} \in \Delta(A_i).
$$

**Coarse correlated equilibrium (CCE).** $\sigma$ may be any joint
distribution. Think of it as a having a shared recommendation device (that global incentive) that hands each agent an action to play before it decides whether to listen. $\sigma$ is a CCE if no agent gains by ignoring the recommendation for one fixed
alternative, chosen in advance:
$$
    E_{a\sim\sigma}[u_i(a)]  \geq E_{a\sim\sigma}[u_i(a_i', a_{-i})]    \quad   \text{for every fixed } a_{i'} \in A_i.
$$

**Correlated equilibrium (CE).** It asks for a stronger requirement: no agent gains
from any *function* of its own recommendation: "whenever I'm told to
play $a_i$, play $\delta(a_i)$ instead":
$$
    E_{a\sim\sigma}[u_i(a)]  \geq  E_{a\sim\sigma}[u_i(\delta_i(a_i), a_{-i})]  \quad   \text{for every } \delta_i: A_i \to A_i
$$

These nest: NE is contained in CE, which is contained in CCE. Nash is the special, hardest case — the one where the recommendation device the agents deem not to deviate and stick to the rationale. 


## Why the obvious fixes all reach for the smallest, hardest set

**Make the sheaf itself game-theoretic.** Recent work casts Nash equilibria
as global sections of a sheaf built from best-response correspondences, with
cohomology measuring the obstruction to a consistent equilibrium existing.
Elegant, and the cohomology-as-obstruction idea is the right shape. But it's
an existence result in the language of derived sheaves and topoi, like Jeffrey likes. It proves sections exist, it doesn't hand you an update rule two neighboring agents can
run with a dot product. In more detail, [Hernández & Sánchez-Soto](https://arxiv.org/abs/2606.01663) define a sheaf whose stalks contain utility vectors and strategy distributions, with restriction maps encoding both geometric parallel transport and best-response dynamics. However, their framework *is not* differentiable, as far as I understand.

**Learn it with a big neural net.** [Neural Equilibrium Solvers](https://arxiv.org/abs/2210.09257) take an entire payoff tensor and output an equilibrium (Nash, CE, or CCE) in one
forward pass, after iterative training over many sampled games. It works (on paper)
but "takes the whole payoff tensor as input" is precisely what a graphical,
locally-communicating architecture is trying to avoid needing.


### Method

The network consumes a 4-channel tensor of shape [B, 4, N, A1, ..., AN]:
- Channel 0: Normalized payoffs `Ĝ`
- Channel 1: Target epsilon `ε̂` (broadcasted)
- Channel 2: Target joint strategy `σ̂` (broadcasted)
- Channel 3: Welfare `W` (broadcasted)

The pipeline is following

```
Input: [B, 4, N, A1, ..., AN]
    ↓
[1] EquivariantPayoffToPayoff × K   (maintains joint action space)
    ↓
[2] PayoffsToDuals                  (marginalizes to player-action space)
    ↓
[3] EquivariantDualToDual × L       (refines in individual space)
    ↓
[4] Final projection + Softplus     (ensures α ≥ 0)
Output: α — [B, 1, N, A] for CCE, [B, 1, N, A, A] for CE
```

Given duals α, the closed-form primal recovery implements Equation (7) from the paper [1]:
1. Compute deviation contributions per player (`gain_p`)
2. Form logits: `l(a) = μ·W(a) − ·Σ_p gain_p(a)`
3. Recover `σ(a) ∝ σ̂(a) · exp(l(a))`
4. Recover `ε_p = (ε̂_p − ε⁺) · exp(−Σα_p / ρ) + ε⁺`


The dual loss is minimised:
```
L_dual = log_sum_exp + ε⁺ · Σ_p Σ_a α_p(a) − ρ · Σ_p ε_p
```

**Turn the dual into a payment.** This is the closest of the three, described in the [paper](https://arxiv.org/abs/2606.20960), so it's worth writing down exactly where it breaks. Suppose agent $i$'s whole decision is a scalar $\pi_i \in [0,1]$ (probability of the "compliant" action), and it receives a transfer $p_i$ in expectation, conditional on realising the compliant action, forfeited otherwise. Its augmented objective is

$$
    \hat{U}_i(\pi_i) = U_i(\pi_i, \pi_{-i}) + \pi_i  p_i
$$

Best-response ascent moves $\pi_i$ along $d(\hat{U}_i)/d(\pi_i) = dU_i/d(\pi_i) + p_i$.
A natural dual-ascent rule for the payment, driven by "does agent i still
want to deviate," takes the form

$$
\begin{aligned}
    p_i  \leftarrow  [ p_i + \beta \cdot \text{deviation\_gain}_i ]_+ ,   \\
    \text{deviation\_gain}_i = -dU_i/d(\pi_i) - p_i
\end{aligned}
$$

Look at those last two equations together: deviation_gain_i and
$\text{deviation\_gain}_i$ are the same quantity up to sign. The instant
$\text{deviation\_gain}_i$ hits zero, the primal gradient is *also* exactly zero. Both variables freeze at precisely the point of indifference, never at a point of actual preference because one residual is driving both updates at once. This might cause a stalling problem where the agents stuck in a "half cooperation state".


> All three fixes are aimed at converging to the Nash equilibrium. That's the thing worth questioning. Why?
{: .prompt-warning }

## Two theorems that say what target to actually aim at

**Definition (uncoupled dynamics).** A dynamic is a family of update rules
$\phi_i$, one per agent, where $\phi_i$ depends on the history of play and, at
most, agent i's own payoff function $u_i$, never on $u_j$ for $j \neq i$.

**Theorem (Hart & Mas-Colell, 2003).** There is no uncoupled dynamic that
is guaranteed to converge to a Nash equilibrium of every game, even when
that game has a unique Nash equilibrium.

Sheaf-ADMM, and every purely local extension of it, including the payments
attempt above, is **uncoupled** by construction. Nash was never the right
target for this class of system. Not an engineering shortcoming. [^2]

**Definition (regret).** After T rounds of play, agent i's average regret
for an alternative action a_{i'} is

$$
    R_i^T(a_{i'})  =  \frac{1}{T} \sum_{t=1}^T [ u_i(a_{i'}, a_{-i}^t) - u_i(a_i^t, a_{-i}^t) ]
$$

An algorithm has no (external) regret if $\max_{a_{i'}} R_i^T(a_{i'}) \to 0$ as
$T \to \infty$, against any sequence of opponents' play.

**Theorem (folk result; Hart & Mas-Colell, 2000).** If every agent runs a
no-external-regret algorithm (e.g. Hedge / multiplicative weights), the
empirical distribution of joint play converges to the coarse correlated
equilibrium set. If every agent achieves the stronger no-internal-regret
(no-swappregret) condition (regret matching is the canonical example)
the empirical distribution converges to the correlated equilibrium set.

Both of these are uncoupled. Both carry an unconditional guarantee, for
every game, with no hand-tuned margin required. Given the equilibria containment (NE subset CE subset CCE), this isn't a downgrade in ambition, it's aiming at exactly the set the architecture can actually be guaranteed to hit.

## Finding a place in the architecture

**The sheaf condition is reattached to belief about the correlation device.** As for each agent $i$ its policy $\pi_i$ is private, so nobody else needs to agree with it, so forcing $F_{ij}(pi_i) = F_{ji}(pi_j)$ on raw strategies doesn't correspond to any meaningful incentive drive. But CE and CCE are defined against a shared correlation device $\sigma$, and if $\sigma$ respects the graph (we can assume it quaite naturally) every edge $(i,j)$ has a genuine shared object: the edge marginal $\sigma_{ij}$. Under partial observability (the actual reason, I guess, why Sheaf-ADMM's restriction maps are learned rather than raw broadcast), $i$'s running estimate of $\sigma_{ij}$ and $j$'s running estimate of it can disagree.

That is exactly a sheaf consensus problem
$$
    z_ij  =  F_{i\to(i,j)}(x_i)  =  F_{j\to(i,j)}(x_j)
$$

reconciled by the same $z$/$u$ loop Sheaf-ADMM already has

$$
\begin{aligned}
    z_ij  \leftarrow  \text{average}( F_{i\to(i,j)}(x_i) + u_i ,  F_{j\to(i,j)}(x_j) + u_j ), \\
    u_i   \leftarrow  u_i + ( F_{i\to(i,j)}(x_i) - z_{ij} ), \\
    u_j   \leftarrow   u_j + ( F_{j\to(i,j)}(x_j) - z_{ij} ).
\end{aligned}
$$

It is just aimed at beliefs about the recommendation marginal instead of raw state. This earns its keep rather than being restored for symmetry: once $i$ has a reconciled $z_ij$, it can compute a *conditional* counterfactual: "given I was recommended $a_i$, would $a'$ have done better specifically then" which is swap regret rather than external regret, and swap regret isexactly what upgrades the guarantee from CCE to CE

$$
    R_i[a_i \to a']  \leftarrow  R_i[a_i \to a']
        +  E_{a_j\sim z_{ij}(\cdot|a_i)}[ u_i^{(i,j)}(a', a_j) ]  -  E_{a_j\sim z_ij(\cdot|a_i)}[ u_i^{(i,j)}(a_i,a_j) ]
$$


> HOWEVER, If observability is already full ($i$ sees $j$'s literal action every round), both sides' estimate of the correlation device is already the sameempirical distribution and the sheaf update won't really do anything. Moreover, we need to keep the regret estimation quite accurate to maintain sheaves' updates bounded (it's necessary to prove!)
{: .prompt-warning }

**A bounded constant instead of unbounded regret.** We might partly address the latter problem. Extental regret R_i^T is an average, so it stays bounded by the payoff range no matter how large $T$ gets. What does grow without bound is the raw running sum behind it, before dividing by $T$. To alleviate that, one can propose a single per-edge scalar, no growing history, a fixed willingness-to-pay-style cap. *Or just introduce a discount factor to the updates*

$$
    R_i[a']  \leftarrow \gamma R_i[a']  +  (1 - \gamma) ( \hat{u}_i(a') - \hat{u}_i(a_i^t) )
$$

for a fixed $\gamma \in (0,1)$. This gives $O(1)$ memory (or not...). It isn't free: the classical guarantee ("empirical play converges to the CE/CCE of the game") no longer applies as stated. What replaces it is convergence toward the CE/CCE of the *recent* game — the right notion if the underlying game is non-stationary (plausible here, since the game is being learned jointly with the architecture), but a different, much weaker result.


## Brigded algorithm sketch

The corrected algorithm

```
    ALGORITHM (edge consensus + bounded swap-regret)
    ------------------------------------------------------------
    Initialize R_i[a_i -> a'] <- 0  for every agent i, a_i, a' in A_i.
    Initialize z_ij, u_i         arbitrarily, one per edge.

    For t = 1, 2, ..., T:
      1. Each i draws a_i^t ~ pi_i^t,   pi_i^t(a') ∝ exp( eta * R_i[a_i^{t-1} -> a'] )
      2. Full observability: i observes a_j^t directly for j in N(i).
         Partial observability: skip to step 3.
      3. (fast inner loop, partial-observability case only) reconcile edge belief:
             z_ij <- average( F_{i->(i,j)}(x_i)+u_i , F_{j->(i,j)}(x_j)+u_j )
             u_i  <- u_i + ( F_{i->(i,j)}(x_i) - z_ij )
             u_j  <- u_j + ( F_{j->(i,j)}(x_j) - z_ij )
      4. Each i computes, for every a' in A_i:
             uhat_i(a_i^t -> a') = sum_{j in N(i)}  E_{a_j~z_ij(.|a_i^t)}[ u_i^{(i,j)}(a',a_j) ]
             (full observability: replace the expectation with the literal a_j^t)
      5. Discounted swap-regret update:
             R_i[a_i^t -> a']  <-  gamma*R_i[a_i^t->a'] + (1-gamma)*( uhat_i(a_i^t->a') - uhat_i(a_i^t->a_i^t) )

    *Output*: exponentially-weighted empirical play (or plain time-average if gamma=1).

    *Guarantee*: full observability and gamma=1 (no discount) recovers
    the CE. Partial observability additionally needs the two-timescale assumption (previous scetion). Discounting (gamma<1) trades that guarantee for the weaker, adaptive/tracking version.
```

## The end

The algorithm in is a sketch, not a result. What it's pointing at is
two literatures that don't currently cite each other: differentiable
distributed sheaf architectures, and several decades of no-regret learning
theory. Looking, from here, like they're aimed at the same spot from two
different directions.

> If someone wants to actually build it, I'd like to hear about it.
{: .prompt-info }



[^1]: It reminded me of how in a  multi-agent reinforcement learning (MARL), namely in [COMA](https://arxiv.org/abs/1705.08926), the credit assignment problem is solved by calculating a difference reward. It measures an individual agent's unique contribution by taking the global reward and subtracting the reward achieved if that agent had taken a different action.

[^2] There is also a version for MARL algorithms, see this [paper](https://arxiv.org/abs/2206.10614).