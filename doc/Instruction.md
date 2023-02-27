# Instruction on Our Work

## Contents
* Abstract
* Example
* Framework
* Video

## Abstract
> This paper investigates the multi-robot patrolling (MuRP) problem with the objective of approaching a uniform area coverage frequency. The problem requires coordinating a robot team to persistently monitor a given topological environment. Prevailing MuRP solutions for uniform coverage either incur high (non-polynomial) computational complexity operations for the global optimal solution, or recourse to simple but effective heuristics for approximate solutions without any performance guarantee. In this paper, we bridge the gap by proposing an efficient iterative algorithm, namely Entropy Maximized Patroller (EM-Patroller), with the per-iteration performance improvement guarantee and polynomial computational complexity. We reformulate the multi-robot patrolling problem in topological environments as a joint steady state distribution entropy maximization problem, and employ multi-layer perceptron (MLP) to model the relationship between each robot's patrolling strategy and the individual steady state distribution. Then, we derive a multi-agent model-based policy gradient method to gradually update the robots' patrolling strategies towards the optimum. Complexity analysis indicates the polynomial computational complexity of EM-Patroller, and we also show that EM-Patroller has additional benefits of catering to miscellaneous user-defined joint steady state distributions and incorporating other objectives, e.g., entropy maximization of individual steady state distribution, into the objective. We compare EM-Patroller with state-of-the-art MuRP algorithms in a range of canonical multi-robot patrolling environments, and deploy it to a real multi-robot system for patrolling in a self-constructed indoor environment.

In short, we proposed:
* An algorithm that produces stochastic MuRP strategies aiming at approximating uniform coverage by entropy maximization (EM-Patroller), which gives consideration to both computational complexity and performance guarantee.
* Three variants of EM-Patroller, which are:
  - Robost EM-Patroller that also maximizes the average entropy of coverage of individuals; (Robustness)
  - Variational EM-Patroller that approximates arbitrary distributions with KLD minimization; (Flexibility)
  - Soft EM-Patroller that maximizes the average entropy rate of individual policies. (Unpredictability)

## Contribution
* we propose EM-Patroller, which serves as a polynomial complexity algorithm with the per-iteration performance improvement guarantee.
* EM-Patroller has the flexibility of catering to miscellaneous user-defined target joint steady state distribution instead of confining itself to joint steady state entropy maximization.
* EM-Patroller exhibits great robustness performance against individual robot failures when we incorporate the individual steady state distribution entropy maximization into the optimization objective.

## Example
In this section, we provide a simple yet illustrative example to show that reformulating multi-robot uniform patrolling problem as multiple travelling salesmen problem (mTSP) that searches deterministic Hamiltonian paths (espeically cycles) does not guarantee the existence of optimal solutions as well as the superiority of its suboptimal solutions over stochastic strategies, no matter within global or local scopes. Consider an environment with following topology:

<p align="center">
  <img width="400" height="300" src="/doc/figure/Example.png">
</p>

This is the HOUSE environment with slight modification. The topology consists two loops containing 5 and 4 loops respectively and node 3 serves as a pivot node. Global cyclic solution will make node 3 a hotspot thus breaks uniformity, and local solutions (after territory partitioning) does not guarantee the uniformity and depends on the number of robots. In Comparison, our EM-Patroller solves this problem with a Markov chain-based point of view and can approach best uniformity under any given topologies.

## Framework
<p align="center">
  <img width="952" height="350" src="/doc/figure/Framework.jpg">
</p>


