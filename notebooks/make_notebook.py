import nbformat as nbf
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
 md("# Hospital 30-day readmission - training notebook\nRuns the same code as `python -m src.train`. Set `RETRAIN=True` to retrain here (~2 min); otherwise it reads saved artifacts."),
 code("import sys, json; sys.path.insert(0, '..')\nimport pandas as pd, matplotlib.pyplot as plt\nfrom IPython.display import Image, display\nfrom src import config as C\nRETRAIN = False\nif RETRAIN:\n    from src.train import run; run()\nart = C.artifact_dir()\nm = json.load(open(art/'metrics.json'))"),
 md("## 1. Data after cleaning (GST fixed, diagnosis mapped, history features, 45-day censoring)"),
 code("df = pd.read_csv(C.data_dir()/'modeling_table.csv', parse_dates=['admission_date'])\nprint(df.shape, 'prevalence', round(df[C.TARGET].mean(), 3))\ndf.head()"),
 code("df.isna().mean().round(3)[lambda s: s > 0]"),
 md("## 2. Time-based split\nTrain < validation < test by admission date (no shuffling)."),
 code("print(m['n_rows'], m['time_ranges'])"),
 md("## 3. Four models + dummy (validation)"),
 code("pd.read_csv(art/'model_comparison.csv').round(4)"),
 md("## 4. Tuned model, test metrics and success criteria"),
 code("print('best model:', m['best_model']); print('best params:', m['best_params'])\nprint('test:', {k: round(v, 3) for k, v in m['test'].items()})\nprint('rule baseline PR-AUC:', round(m['baseline_rule_test']['pr_auc'], 3), '| untuned LR PR-AUC:', round(m['baseline_logistic_test']['pr_auc'], 3))\nprint('improvement vs baseline: %.0f%% (met=%s)' % (m['improvement_over_baseline']*100, m['success_improvement_met']))\nprint('latency incl. SHAP: %.0f ms (met=%s)' % (m['inference_latency_ms_with_shap'], m['success_latency_met']))"),
 md("## 5. Cost-based threshold\nFN = INR 40,000, FP = INR 1,500, TP = INR 29,500 (follow-up + the 70% it fails to prevent), TN = 0."),
 code("c = pd.read_csv(art/'threshold_curve.csv')\nplt.plot(c.threshold, c.total_cost_inr/1e6); plt.axvline(m['test']['threshold'], ls='--', c='r')\nplt.xlabel('threshold'); plt.ylabel('validation cost (INR million)'); plt.title('Total cost vs threshold'); plt.show()\nprint(m['business'])"),
 md("## 6. SHAP - global explanation"),
 code("display(Image(filename=str(art/'shap_summary.png')))\npd.read_csv(art/'shap_importance.csv').head(10)"),
]
nbf.write(nb, "training.ipynb")
