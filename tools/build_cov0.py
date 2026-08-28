"""Rebuild the feature matrix with no coverage filter, for the sweep.

Writes data/features_cov0.parquet -- deliberately NOT the production
data/features.parquet, which stays exactly as train.py left it. One build at
min_coverage=0 serves every threshold, because build_all computes the coverage
column and the filter is applied afterwards.
"""
import sys, time
sys.path.insert(0, ".")
from valwr import config
from valwr.features import build as fb
from valwr.model import split
from valwr.store import schema

s = config.load(require_key=False)
conn = schema.connect(s.database_path)
b = split.compute(conn)
print(f"norms_as_of={b.train_end}  ({b.n_matches:,} resolved matches)", flush=True)

t = time.time()
rows = fb.build_all(conn, norms_as_of=b.train_end, min_coverage=0)
df = fb.to_frame(rows)
out = s.database_path.parent / "features_cov0.parquet"
df.to_parquet(out, index=False)
print(f"wrote {out}  {len(df):,} rows in {time.time()-t:.0f}s", flush=True)
print(df["coverage"].value_counts().sort_index().to_string(), flush=True)
