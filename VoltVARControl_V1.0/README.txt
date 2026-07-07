Updated Case 1 repo (with Case 2 eval tools + IEEE-14 + robust 123 MAT loader).

Run order (typical):
1) python train_DDPG.py --case 13 --preset fast
2) python train_DDPG.py --case 123 --preset fast
3) python eval_paper_protocol.py --case 13  --trials 10 --dq_max 0.1 --alpha 5 --h 1 --eta_over_sbar 0.01 > results/metrics_13.json
4) python eval_paper_protocol.py --case 123 --trials 10 --dq_max 0.1 --alpha 5 --h 1 --eta_over_sbar 0.01 > results/metrics_123.json
5) python IEEE_13_3p.py
6) python rescore_playback_with_paper_objective.py --in_csv results/trajectory_doe_ieee13_stable.csv --out_csv results/trajectory_doe_ieee13_scored.csv
7) python compare_tables.py --yours13 results/trajectory_doe_ieee13_scored.csv --yours123 results/metrics_123.json --out results/comparison_table.csv --paper
