#!/bin/bash
cd /Users/ashwin/newproject/followup_hi_be_tn_clissta_exp
python evaluate_indic_clitssa.py --languages hi bn ta --tasks xnli qa --n_samples 100 --k 3 --output results/full_indic_evaluation.json
