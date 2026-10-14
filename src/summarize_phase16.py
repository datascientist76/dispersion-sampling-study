"""Summarize Paper 1 Phase-16 mechanism output per kernel and clip value (read-only)."""
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
print("columns:", list(df.columns))
need = {"log_clip", "kernel", "step", "clipped_fraction_cumulative", "max_abs_unclipped_log_ratio"}
missing = need - set(df.columns)
if missing:
    sys.exit(f"Missing columns {missing}. Paste the column list above and I will adapt this.")

last = df[df["step"] == df["step"].max()]
per_kernel = last.groupby(["kernel", "log_clip"])["clipped_fraction_cumulative"].mean()
mx = df.groupby(["kernel", "log_clip"])["max_abs_unclipped_log_ratio"].max()
out = pd.concat([per_kernel.rename("clip_fraction"), mx.rename("max_abs_log_ratio")], axis=1).reset_index()
out["can_clip"] = out["max_abs_log_ratio"] > out["log_clip"]
print(out.to_string(index=False))

print("\nPooled over kernels (compare with 0.118454 at L=2 and 0.021391 at L=6):")
print(last.groupby("log_clip")["clipped_fraction_cumulative"].mean().to_string())
out.to_csv("results/phase0_from_phase16.csv", index=False)
print("\nSaved results/phase0_from_phase16.csv")
