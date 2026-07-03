import numpy as np, rugby_elo as re

df = re.load_results("data/rugby_results.csv")

print("HFA sweep at K=41, eval 2000+:")
for hfa in range(60, 201, 10):
    pred, _, _ = re.run_elo(df, re.EloConfig(k=41, hfa=hfa))
    m = re.evaluate(pred, start_year=2000)
    print(f"  HFA={hfa:3d}  log_loss={m['log_loss']:.4f}  "
          f"brier={m['brier']:.4f}  acc={m['accuracy']:.3f}")

print("\nJoint fine search K x HFA (eval 2000+):")
best = None
for k in range(30, 56, 2):
    for hfa in range(80, 171, 10):
        pred, _, _ = re.run_elo(df, re.EloConfig(k=k, hfa=hfa))
        m = re.evaluate(pred, start_year=2000)
        if best is None or m["log_loss"] < best[0]:
            best = (m["log_loss"], k, hfa, m["brier"], m["accuracy"])
print(f"  best log_loss={best[0]:.4f} at K={best[1]}, HFA={best[2]} "
      f"(brier={best[3]:.4f}, acc={best[4]:.3f})")
