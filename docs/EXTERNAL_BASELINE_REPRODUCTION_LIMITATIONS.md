# External published-baseline comparison: what this workflow does and does not prove

This workflow makes the current comparison reproducible on GitHub-hosted compute and stores the new run's CSVs, report, and figures as an artifact. It does not overwrite historical result folders.

## Run it

1. Open the repository's **Actions** tab.
2. Select **External-baseline comparison (reproducible run)**.
3. Select `smoke` to validate the runner or `full` to execute a matched-seed experiment.
4. Download the generated artifact from the completed workflow run.

## Critical scientific limitation

The currently available external-policy code is **paper-inspired proxy code**, not a faithful reproduction of the published algorithms. The simulation uses scalar communication-rate and sensing-information-gain metrics. It does not implement the native physical-layer models needed to claim numerical reproduction of:

- Elfiatoure et al.'s max-min spectral-efficiency optimization under MASR constraints;
- Özbek and Cumalı's sensing-SNR maximization under communication-SINR and serving-AP constraints;
- Sun et al.'s exact adaptive-hysteresis and kinematic-clustering rules;
- Ammar et al.'s trained POMDP handoff policy.

Consequently, any plots produced by this workflow must retain the label `PAPER_INSPIRED_PROXY`. Do not present them as faithful reproductions or compare their numeric values directly with figures in the papers. A smoke run verifies that the code executes; it is not scientific validation. A full run provides repeatable in-simulator evidence under matched seeds, not proof of exact paper reproduction.

## Research sources

- Elfiatoure et al., *Cell-Free Massive MIMO for ISAC: Access Point Operation Mode Selection and Power Control*, arXiv:2310.09032, https://arxiv.org/abs/2310.09032
- Cumalı and Özbek, *Joint mode selection and user association for ISAC-enabled cell-free MIMO networks*, Physical Communication 77 (2026), https://doi.org/10.1016/j.phycom.2026.103132
- Sun et al., *Mobility Management Framework for Cooperative Cell-Free ISAC Systems*, IEEE Internet of Things Journal (2025), https://doi.org/10.1109/JIOT.2025.3604762
- Ammar et al., *Handoffs in User-Centric Cell-Free MIMO Networks: A POMDP Framework*, IEEE Transactions on Wireless Communications (2024), https://doi.org/10.1109/TWC.2024.3371379

## Next fidelity work

For a defensible paper-to-paper result, the next phase is to reproduce at least one compatible paper's native channel model, objective, constraints, and parameter table, then evaluate the proposed method in that same model. The current scalar simulator cannot establish that by relabeling its existing proxy algorithms.
