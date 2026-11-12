"""Hand-over to other analysis software: MATLAB (.mat), R, Python, SPSS/Stata/Origin/Tableau/Power BI (via CSV or Parquet), Excel."""
import io, re, numpy as np, pandas as pd
from scipy.io import savemat

EXCEL_MAX_ROWS = 1_048_576

def _var(name):
    v = re.sub(r"\W", "_", str(name)).strip("_") or "v"
    return ("t_" + v if v[0].isdigit() else v)[:31]

def table_to_struct(df):
    """DataFrame -> dict of column arrays: numbers as double, text as cell arrays (MATLAB struct of columns)."""
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s): out[_var(c)] = s.to_numpy(float)
        elif pd.api.types.is_numeric_dtype(s): out[_var(c)] = s.to_numpy(float)
        else: out[_var(c)] = np.array(s.astype(str).tolist(), dtype=object)
    return out

def tables_to_mat(tabs, path_or_buf):
    """One struct per table, e.g. raw_runs.post_change_f1, summary.model. Open in MATLAB with S = load('results.mat')."""
    savemat(path_or_buf, {_var(k): table_to_struct(df) for k, df in tabs.items() if len(df) or k == "raw_runs"}, do_compression=True)

def dataset_to_mat(df, path_or_buf):
    """Long-format dataset -> struct of columns; channel columns stay as vectors of length rows."""
    savemat(path_or_buf, {"dataset": table_to_struct(df)}, do_compression=True)

def dataset_bytes(df, fmt):
    """fmt in {'csv','mat','parquet'}; returns (bytes, filename suffix, mime)."""
    if fmt == "csv": return df.to_csv(index=False).encode(), "csv", "text/csv"
    b = io.BytesIO()
    if fmt == "mat": dataset_to_mat(df, b); return b.getvalue(), "mat", "application/octet-stream"
    if fmt == "parquet": df.to_parquet(b, index=False); return b.getvalue(), "parquet", "application/octet-stream"
    raise ValueError(fmt)

MATLAB_SCRIPT = """% Load wsnfd results in MATLAB.  Run from the folder that holds this file.
% Option A: native file (recommended)
S = load('results.mat');                 % one struct per table: S.raw_runs, S.summary, ...
f1  = S.raw_runs.post_change_f1;         % numeric columns are double vectors
mdl = string(S.raw_runs.model);          % text columns are cell arrays
scn = string(S.raw_runs.scenario);
% Example: mean post-change F1 by model for scenario S2
for m = unique(mdl)'
    fprintf('%s  %.3f\\n', m, mean(f1(mdl==m & scn=="S2"), 'omitnan'));
end
% Option B: any table as CSV
T = readtable(fullfile('tables','summary.csv'));
% Boxplot over seeds
boxplot(f1(scn=="S2"), mdl(scn=="S2")); ylabel('post-change F1');
% Labelled dataset (if you exported dataset_*.mat):  D = load('dataset_S2_seed0.mat').dataset;
"""

R_SCRIPT = """# Load wsnfd results in R.  setwd() to the folder that holds this file.
runs <- read.csv(file.path("tables", "raw_runs.csv"))
summ <- read.csv(file.path("tables", "summary.csv"))
print(aggregate(post_change_f1 ~ model + scenario, runs, mean))
boxplot(post_change_f1 ~ model, data = subset(runs, scenario == "S2"), ylab = "post-change F1")
# Friedman test across models, blocked by seed (scenario S2)
d <- subset(runs, scenario == "S2")
w <- reshape(d[, c("seed", "model", "post_change_f1")], idvar = "seed", timevar = "model", direction = "wide")
print(friedman.test(as.matrix(w[, -1])))
# Labelled dataset:  D <- read.csv("dataset_S2_seed0.csv")   (or arrow::read_parquet)
"""

PY_SCRIPT = """# Load wsnfd results in Python / Jupyter.
import pandas as pd
runs = pd.read_csv("tables/raw_runs.csv")
print(runs.groupby(["scenario", "model"])["post_change_f1"].agg(["mean", "std", "count"]))
# xls = pd.read_excel("results.xlsx", sheet_name=None)   # every sheet as a DataFrame
# D = pd.read_parquet("dataset_S2_seed0.parquet")        # labelled dataset
"""

SCRIPTS = {"load_in_matlab.m": MATLAB_SCRIPT, "load_in_R.R": R_SCRIPT, "load_in_python.py": PY_SCRIPT}

def make_notebook():
    """A ready-to-run Jupyter notebook (JSON string) that loads the exported tables; needs only pandas, numpy, matplotlib, scipy."""
    import json
    md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
    code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}
    cells = [
        md("# wsnfd results notebook\nRun this from the folder you unzipped the export bundle into. Every cell uses the exported tables, so it works without the wsnfd package."),
        code("import pandas as pd, numpy as np, matplotlib.pyplot as plt\nfrom scipy import stats\npd.set_option('display.width', 140)\nruns = pd.read_csv('tables/raw_runs.csv')\nruns.head()"),
        md("## Summary by scenario and model"),
        code("measure = 'post_change_f1'   # change to any numeric column of runs\ng = runs.groupby(['scenario', 'model'])[measure]\nsumm = g.agg(['mean', 'std', 'count'])\nsumm['ci95'] = 1.96 * summ['std'] / np.sqrt(summ['count'])\nsumm.round(4)"),
        md("## Distribution over seeds"),
        code("sc = sorted(runs.scenario.unique())[-1]   # pick a scenario\nd = runs[runs.scenario == sc]\nmodels = sorted(d.model.unique())\nplt.boxplot([d[d.model == m][measure].dropna() for m in models])\nplt.xticks(range(1, len(models) + 1), models)\nplt.ylabel(measure); plt.title(f'Scenario {sc}'); plt.show()"),
        md("## Block-F1 over time (if blocks.csv is present)"),
        code("import os\nbp = 'tables/block_curves.csv'\nif os.path.exists(bp):\n    b = pd.read_csv(bp); print(b.columns.tolist())\nelse:\n    print('No block curves table in this bundle; load blocks.csv from a results folder instead.')"),
        md("## Paired test between two models (same seeds)\nWilcoxon signed-rank on matched seeds. With few seeds the p-value has a floor, so report the effect size too."),
        code("a, b_ = 'M1', 'M3'\nw = d.pivot_table(index='seed', columns='model', values=measure)\nif {a, b_} <= set(w.columns):\n    x, y = w[a].dropna(), w[b_].dropna(); i = x.index.intersection(y.index); diff = (y[i] - x[i])\n    print('n pairs', len(i), 'mean diff', round(diff.mean(), 4), 'dz', round(diff.mean() / diff.std(ddof=1), 2))\n    print(stats.wilcoxon(x[i], y[i]) if len(i) > 5 else 'Too few pairs for a Wilcoxon test.')"),
        md("## Labelled dataset (optional)\nExport one from the workbench's Data tab (CSV or Parquet) or `python -m wsnfd.export`."),
        code("# D = pd.read_parquet('dataset_S2_seed0.parquet')   # or pd.read_csv('dataset_S2_seed0.csv')\n# D.groupby(['segment', 'fault_type']).size()"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(nb, indent=1)
