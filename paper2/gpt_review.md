# External review of paper 2 (GPT-4.1, requested by author)

**1. Summary**

This paper systematically investigates the limits of “scramble inversion” (training a denoiser to invert random walks from the goal, then acting by reverse-process rollout) as a general approach to solving sequential decision problems. The author constructs a “ten-domain assumption ladder,” where each environment removes one structural property of the Rubik’s Cube (where scramble inversion is known to excel), and compares scramble inversion to deep approximate value iteration (DAVI) under matched architectures and compute. The study finds that scramble inversion is dominant in large, deterministic, reversible domains, but fails or requires adaptation in settings with irreversible actions, continuous actions, stochasticity, or extreme horizons. The paper provides a detailed empirical map of where the method works, where it fails, and why.

---

**2. Strengths**

- **Systematic, Mechanistic Study:** The paper is unusually thorough, constructing a clear “assumption ladder” with ten domains, each carefully designed to isolate the effect of removing one property. This is a rare and valuable approach in empirical RL research.
- **Matched Protocols and Controls:** The use of identical architectures, compute budgets, and careful validation (including exact oracles and replay verification) ensures that observed differences are due to the learning objective, not confounds.
- **Sharp Empirical Findings:** The results are clear-cut, with large, reproducible performance differences (e.g., 100% vs. 0.6% on the 24-puzzle), and the analysis is mechanistic rather than anecdotal.
- **Constructive Failure Analysis:** The paper not only identifies failure modes (e.g., deadlock-blindness, regression collapse, schedule coverage) but also proposes and tests concrete fixes (hybrid value veto, distributional heads).
- **Reproducibility and Open Science:** All code, environments, and logs are released, with validation gates for correctness.
- **Clarity of Practitioner Guidance:** The “checklist” at the end is practical and actionable.

---

**3. Weaknesses (ranked by severity)**

1. **Limited Generalization to Large-Scale, Realistic RL Domains:** The study is restricted to small, synthetic, or classical planning environments. There is no evidence that scramble inversion scales to high-dimensional, partially observable, or visually complex RL tasks (e.g., Atari, MuJoCo, robotics). This limits the practical impact.
2. **Stochasticity Boundary is Underexplored:** The failure on 2048 is attributed to exogenous randomness, but the paper does not attempt any stochastic-aware reverse process or discuss possible approaches (e.g., marginalization, stochastic diffusion models). This leaves a major open question unaddressed.
3. **Single Architecture Type (MLP):** Only MLPs are used, even for spatial or visual domains (mazes, Sokoban). This may underestimate the potential of value learning or denoising with more appropriate architectures (e.g., CNNs, GNNs).
4. **Limited Search at Evaluation:** Only greedy or 1-step lookahead is used; stronger search (beam, MCTS) could change the relative performance, especially for value-based methods.
5. **Small Number of Seeds and Hardware:** Only three seeds per configuration (one for 24-puzzle), and all experiments on a single consumer GPU. This raises questions about robustness and generality.
6. **Some Load-Bearing Figures Not Visible:** The main claims about noising/denoising behavior and failure modes rely on figures (3, 4, 5, 6) that are not accessible in this review. While the descriptions are detailed, the absence of visual evidence is a concern.
7. **Lack of Theoretical Analysis:** The study is almost entirely empirical; there is little formal analysis of why certain boundaries exist (e.g., why schedule coverage is limiting, or why stochasticity is fatal).

---

**4. Questions for the Authors**

- Can scramble inversion be extended to handle exogenous stochasticity (e.g., by modeling the reverse process as a conditional distribution over possible predecessor states)? Have you attempted this or do you see a principled path forward?
- How sensitive are your results to the architecture choice? Would CNNs or GNNs improve value learning or denoising in spatial domains?
- For domains where the value baseline underperforms (e.g., 24-puzzle), is this due to the value function’s inability to propagate over long horizons, or could search (e.g., MCTS) close the gap?
- In the hybrid (denoiser + value veto) approach, how is λ selected, and how robust is performance to this hyperparameter?
- How does the method fare in domains with both partial observability and stochasticity (e.g., POMDPs with random transitions)?
- Did you attempt any ablations on the noising schedule (e.g., stratified sampling, deeper walks) to improve coverage in Hanoi or similar domains?

---

**5. Missing Experiments or Analyses**

- **Larger-Scale or Visual RL Domains:** At least one experiment on a more realistic RL domain (e.g., Atari, MuJoCo, or a visual navigation task) would greatly strengthen the paper’s claims about generality.
- **Stochastic Reverse Process:** An attempt to implement a stochastic-aware reverse process (even a toy version) would clarify whether the stochasticity boundary is fundamental.
- **Architecture Ablations:** Comparing MLPs to CNNs or GNNs on spatial tasks would test whether the observed gaps are due to the learning objective or the architecture.
- **Search at Evaluation:** Including results with beam search or MCTS for both methods would provide a fairer comparison, especially in hard domains.
- **More Seeds and Hardware Diversity:** Running more seeds and on different hardware would increase confidence in robustness.
- **Theoretical Analysis:** Even a brief formal discussion of why schedule coverage or stochasticity are hard boundaries would be valuable.

---

**6. Writing/Presentation Issues**

- **Dense, Jargon-Heavy Prose:** The writing is concise but often cryptic, with heavy use of technical terms and shorthand (e.g., “deadlock-blind,” “shell coverage,” “Bayes-optimal denoiser spreads mass over that set”) that may be inaccessible to non-experts.
- **Figures are Load-Bearing:** The main mechanistic claims rely on figures that are not visible; the text descriptions are good, but the paper would be much weaker without the visuals.
- **Table 1 Notation:** The use of symbols (✓, △, ✗) is clear, but the ranges and notation (e.g., 65.2%(64.9 − −65.5)) could be explained more explicitly.
- **References to Companion Paper:** Several key concepts (e.g., shell coverage, schedule analysis) are deferred to [8], which is not public. The paper should be more self-contained.
- **Minor Typos:** A few minor typographical errors (e.g., “every RL problem can be a diffusion problem”?; “mechanical replay verification of every claimed solution”) but nothing major.

---

**7. Overall Recommendation**

**Recommendation:** Major Revision  
**Score:** 6/10  
**Confidence:** 8/10

**Justification:**  
This is a rigorous, systematic, and insightful empirical study that significantly clarifies the boundaries of scramble inversion as a general RL method. The experimental design is exemplary, and the findings are sharp and actionable. However, the scope is limited to small, synthetic domains, and the most important open question—handling stochasticity—is left unaddressed. The paper would be much stronger with at least one experiment in a more realistic RL domain, an attempt at a stochastic-aware reverse process, and some architectural/search ablations. The writing, while precise, is dense and relies heavily on figures and a companion paper. Overall, this is a valuable contribution, but not yet ready for publication at a top venue without substantial additional work.