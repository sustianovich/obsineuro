Study the Posterior Abstention Infrastructure

Project: Gentle RAG / obsidian-rag

Objective:
Understand the current posterior abstention system and decide whether it is safe to enforce a positive threshold. Do not modify files or configuration.

Important:
Distinguish between:
1. Source-code defaults
2. `.env.example` defaults
3. Active local `.env`
4. Observation mode (`enabled=true`, threshold `0.00`)
5. Enforced abstention with a positive threshold

Read in this order:
1. `app/rag/abstention.py` — full file
2. `app/config.py` — Settings fields, `env_bool()`, and env loading
3. `.env.example` and the abstention variables in `.env`
4. `app/rag/retrieval.py` — pre-retrieval and post-fusion integration
5. `scripts/calibrate_threshold.py`
6. `scripts/calibrate_posterior_threshold.py`
7. `evaluations/questions.json`
8. `evaluations/pdpcm_questions.json`
9. `evaluations/pdpcm_abstention_negatives.json`
10. Relevant abstention sections in `README.md`
11. `app/rag/agents.py` — `SUFICIENCIA` and verifier abstention
12. `app/rag/citations.py` and SSE/synchronous integration in `app/main.py`

Investigate:
- What `evaluate_posterior_abstention()` returns when disabled
- Every exact condition producing `should_abstain=True`
- Empty-selection and no-signal behavior
- All six signals, weights, normalization, and reason strings
- What `_branch_agreement()`, `top_margin`, and `fragment_sufficiency` actually measure
- Whether `best_semantic_score` is truly the maximum
- How behavior changes when reranking, graph, FTS5, MMR, or `top_k` changes
- All references to `posterior_abstention_threshold`
- Difference between:
  - `calibrate_threshold.py`, which calibrates `RAG_MIN_SIMILARITY`
  - `calibrate_posterior_threshold.py`, which calibrates the post-fusion score
- Dataset counts and class imbalance
- Whether PDPCM negatives have been reviewed by the domain owner
- Interaction with verifier `SUFICIENCIA`, writer skipping, citations, and SSE
- Whether active configuration is observation-only or enforcing

Safety requirements:
A positive threshold may be recommended only if validation demonstrates:
- Zero false abstentions on answerable PDPCM questions
- Document recall remains 100%
- MRR remains at least 0.938
- Abstention precision/recall are reported
- Results hold on a separate held-out set

Deliverable:
A concise report with:
A. Current/default/active state
B. Threshold configuration
C. Exact abstention triggers
D. Signal weights and calibration status
E. Dataset counts and limitations
F. Integration risks
G. Recommended threshold, or an explicit “insufficient evidence”
H. Exact next command to run

Do not implement changes. Do not claim that `0.35` is calibrated without posterior-score evidence. If distributions overlap, recommend keeping threshold `0.00` and using verifier abstention.