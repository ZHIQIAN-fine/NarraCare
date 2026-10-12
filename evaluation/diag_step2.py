"""Step 2 Diagnosis: Is LinUCB actually learning?

Checks:
- Day 0 vs Day 29 top-5 for ACL motivated user
- A matrix norms for different tasks
- Feature vector samples
"""
import sys, json, logging
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
logging.basicConfig(level=logging.WARNING)

from src.pipeline import NarraCarePipeline

pipe = NarraCarePipeline("config/default.yaml",
                         config_overrides={"recommender": {"algorithm": "linucb"}})

user_ids = pipe.simulator.get_all_user_ids()
for uid in user_ids:
    raw = pipe.simulator.get_profile(uid)
    pipe.memory.set_profile(uid, pipe._normalize_profile(raw))

# Identify ACL motivated users
acl_motivated = []
for uid in user_ids:
    p = pipe.simulator.get_profile(uid)
    if p.get("condition") == "ACL_reconstruction" and p.get("personality") == "motivated":
        acl_motivated.append(uid)

target_uid = acl_motivated[0] if acl_motivated else user_ids[0]
print(f"Target user: {target_uid}")
profile = pipe.simulator.get_profile(target_uid)
print(f"  Condition: {profile.get('condition')}, Personality: {profile.get('personality')}")

# ---------- Day 0 snapshot ----------
state_d0 = pipe.simulator.simulate_daily_state(target_uid, 0)
result_d0 = pipe.run_single_turn(target_uid, state_d0, mode="simulation", day=0)

print("\n=== Day 0 top-5 (before any learning) ===")
# Get top-5 from all candidates
all_cands = list(pipe.task_library)
if pipe.safety_agent.enabled:
    safe_cands = pipe.safety_agent.pre_filter(all_cands, state_d0,
                                               pipe.memory.get_profile(target_uid))
else:
    safe_cands = all_cands
fv_d0 = result_d0["feature_vector"].to_numpy()
top5_d0 = pipe.recommender.recommend(fv_d0, safe_cands, top_k_override=5)
for t in top5_d0:
    print(f"  {t['task_id']} {t['name']:<40s} diff={t.get('difficulty')} cat={t.get('category')} score={t.get('score', 0):.4f}")
print(f"\nFeature vector day 0: {np.array2string(fv_d0, precision=3)}")

# ---------- Run full 30-day simulation ----------
print("\nRunning 30-day simulation...")
logs = pipe.run_simulation(n_days=30)

# ---------- Day 29 snapshot ----------
state_d29 = pipe.simulator.simulate_daily_state(target_uid, 29)
pipe.memory.update_state(target_uid, state_d29)
cr7, ch = pipe._get_memory_features(target_uid)
fv_d29 = pipe.user_understanding.parse_structured(
    state_d29, pipe.memory.get_profile(target_uid),
    completion_rate_7d=cr7, consecutive_high_intensity=ch,
).to_numpy()

print(f"\n=== Day 29 top-5 (after 30 days of learning) ===")
safe_cands_d29 = pipe.safety_agent.pre_filter(all_cands, state_d29,
                                                pipe.memory.get_profile(target_uid)) \
    if pipe.safety_agent.enabled else all_cands
top5_d29 = pipe.recommender.recommend(fv_d29, safe_cands_d29, top_k_override=5)
for t in top5_d29:
    print(f"  {t['task_id']} {t['name']:<40s} diff={t.get('difficulty')} cat={t.get('category')} score={t.get('score', 0):.4f}")
print(f"\nFeature vector day 29: {np.array2string(fv_d29, precision=3)}")

# ---------- A matrix norms ----------
model = pipe.recommender.model  # LinUCBModel
print("\n=== LinUCB A matrix norms (top 15 most-updated arms) ===")
update_counts = {}
for arm_id, A_mat in model.A.items():
    # n_updates ≈ trace(A) - d (since A starts as I_d)
    n_updates_approx = np.trace(A_mat) - model.d
    update_counts[arm_id] = (np.linalg.norm(A_mat - np.eye(model.d), 'fro'), n_updates_approx)

sorted_arms = sorted(update_counts.items(), key=lambda x: x[1][0], reverse=True)[:15]
for arm_id, (norm_val, n_upd) in sorted_arms:
    theta = np.linalg.inv(model.A[arm_id]) @ model.b[arm_id]
    print(f"  {arm_id:<6s}  ||A-I||_F = {norm_val:8.3f}  ~updates = {n_upd:6.1f}  "
          f"||theta|| = {np.linalg.norm(theta):6.4f}  "
          f"theta_mean = {np.mean(theta):+.4f}")

# Check: how many arms were never updated
never_updated = sum(1 for arm_id in model.A
                    if np.allclose(model.A[arm_id], np.eye(model.d)))
total_arms = len(model.A)
print(f"\nTotal arms seen: {total_arms}, Never updated (A==I): {never_updated}")

# ---------- Feature vector diversity ----------
print("\n=== Feature vector samples (5 different users, day 15) ===")
for uid in user_ids[:5]:
    s = pipe.simulator.simulate_daily_state(uid, 15)
    p = pipe.memory.get_profile(uid)
    cr, ch2 = pipe._get_memory_features(uid)
    fv = pipe.user_understanding.parse_structured(s, p, completion_rate_7d=cr, consecutive_high_intensity=ch2)
    print(f"  {uid}: {np.array2string(fv.to_numpy(), precision=3)}")

# ---------- Reward analysis over time ----------
print("\n=== Reward by day (averaged across users) ===")
from collections import defaultdict
day_rewards = defaultdict(list)
for ix in logs:
    day_rewards[ix["day"]].append(ix.get("reward", 0))
for d in [0, 5, 10, 15, 20, 25, 29]:
    vals = day_rewards.get(d, [])
    if vals:
        print(f"  Day {d:2d}: mean_reward = {np.mean(vals):.4f} ± {np.std(vals):.4f}")

print("\nDone.")
