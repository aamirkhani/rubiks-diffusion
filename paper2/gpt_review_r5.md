# GPT-4.1 full-paper review of r5 (2026-08-20, round 2)

**1. Summary**

This paper empirically investigates the applicability and limitations of formulating RL problems as diffusion problems via "scramble inversion"—training a denoiser to invert random-walk noising from the goal and acting via reverse-process rollout. Using a carefully constructed "assumption ladder" of ten diverse domains, each removing a property of the Rubik’s Cube (where this approach excels), the study benchmarks scramble inversion against a matched value-learning baseline (DAVI). The results provide a nuanced map: scramble inversion is highly effective at scale and robust to many variations, but fails fundamentally in the presence of stochastic dynamics and has repairable weaknesses with irreversibility and continuous actions. The paper is thorough, methodologically careful, and addresses all major concerns from the first review round.

---

**2. Overall Score: 8/10 (Minor Revision)**
- **Justification:** The paper is a rigorous, comprehensive, and insightful empirical study with clear contributions and careful experimental design. The revisions address all major prior concerns. Some additional clarifications and minor analyses would further strengthen the work.
- **Decision:** **Accept (after minor revision)**

---

**3. Strengths**

- **Thorough empirical mapping:** The ten-domain "assumption ladder" is well-motivated and covers a broad spectrum of RL problem structures.
- **Matched protocols:** Careful use of matched architectures, compute, and validation protocols ensures fair comparison.
- **Reproducibility:** All code, environments, and logs are released; validation gates for every environment are described.
- **Constructive failure analysis:** Identifies not just where scramble inversion fails, but why, and proposes/validates effective fixes (e.g., hybrid value veto, distributional head).
- **Seed variance reporting:** Now includes 3-seed min-max ranges throughout, addressing a key prior weakness.
- **Mechanistic insight:** The study provides not just performance numbers but mechanistic explanations (e.g., schedule coverage in Hanoi).
- **Clarity of results:** The practitioner’s checklist distills actionable lessons.

---

**4. Weaknesses (ranked by importance)**

1. **Limited search at evaluation:** All results use greedy or 1-step lookahead; the impact of stronger search (e.g., beam search) is not quantified, though it is mentioned as possible.
2. **Single hardware/backbone:** Only MLPs and one GPU platform are used; results may not generalize to more complex architectures or hardware.
3. **Limited domain diversity in some rungs:** Some domains (e.g., Sokoban, Peg Solitaire) are generated from a single family of instances; generalization to broader distributions is not tested.
4. **2048 baseline is heuristic:** The baseline for 2048 is not a learned policy, which limits the strength of the negative result.
5. **Some results single-seed:** The 24-puzzle uses a single seed due to compute, though the gap is large.
6. **Writing density:** The writing is highly compressed and could be more accessible, especially in the background and results sections.
7. **Figure accessibility:** The paper relies on figures for key qualitative insights, but these are not described in enough detail for readers without access to them.

---

**5. Correctness Check**

- **Claims are generally well-supported** by the reported results and experimental design.
- **Seed variance** is now reported throughout, as requested.
- **Failure mode fixes** (Sokoban hybrid, pendulum distributional head) are demonstrated with quantitative improvements.
- **Hanoi analysis** is rigorous; the shell coverage and walk coverage are measured and explained.
- **Stochasticity probe** for 2048 is negative, and the reasoning is sound.
- **No major inconsistencies** or unsupported claims detected.
- **Experimental controls** (e.g., validation gates, replay verification) are robust.

---

**6. Clarity/Presentation Issues**

- **Title, abstract, and body are now coherent** and self-contained, with the new background section.
- **Some sections are dense** (e.g., background, results) and could benefit from more signposting or explanatory text.
- **Figures are referenced but not described in detail**; readers without access to them may miss key qualitative findings.
- **Practitioner’s checklist is excellent** and succinctly summarizes the main lessons.

---

**7. Prioritized, Actionable Revision List**

**(A) Requires only writing/clarification:**
1. **Explicitly discuss beam search:** Add a short quantitative or qualitative discussion of how stronger search (e.g., beam width >1) affects results in at least one domain (even if only in supplementary).
2. **Describe figures more fully:** In the main text or captions, provide more detailed descriptions of what the noising/denoising strips and galleries show, for accessibility.
3. **Clarify 2048 baseline:** Explicitly state that the 2048 baseline is heuristic, and discuss the implications for the strength of the negative result.
4. **Expand on domain generation:** Briefly discuss the limitations of using a single family of Sokoban/Peg Solitaire instances and possible effects on generalization.
5. **Highlight seed count for 24-puzzle:** In the results, clearly state that the 24-puzzle uses a single seed and justify why the result is robust despite this.
6. **Add signposting in dense sections:** Consider adding more topic sentences or summary statements in the background and results sections to aid readability.

**(B) Optional (would further strengthen, but not required for acceptance):**
7. **(Optional) Provide a small beam search ablation:** If feasible, add a table or figure showing the effect of beam width on solve rates in one or two domains.
8. **(Optional) Test with a learned 2048 baseline:** If possible, compare to a simple learned value or policy baseline for 2048, to further support the stochasticity boundary claim.

---

**8. Questions for the Authors**

1. **Beam search:** Did you run any experiments with beam search or other non-greedy solvers at evaluation? If so, how do results change, especially in domains where greedy rollout is suboptimal?
2. **Domain generalization:** How sensitive are the results in Sokoban and Peg Solitaire to the choice of instance generator? Have you tested on out-of-distribution or more diverse instances?
3. **Continuous action discretization:** For pendulum and Mountain Car, how was the discretization chosen (21 bins, 3 bins)? Did you test sensitivity to this choice?
4. **Stochasticity boundary:** Could you speculate on what a "stochastic-aware" reverse process for 2048 might look like, or whether there are tractable ways to approximate it?
5. **Schedule design:** For domains like Hanoi, did you consider alternative noising processes (e.g., with explicit mixing or restarts) to overcome the shell coverage limitation?
6. **Resource usage:** For the 24-puzzle, given the single-seed result, do you have any evidence (e.g., from partial runs or smaller depths) that the result is stable across seeds?

---

**Overall:**  
This is an excellent and timely empirical study of a hot topic. The revisions are substantial and address all major prior concerns. With minor clarifications and some expanded discussion, the paper is ready for acceptance.