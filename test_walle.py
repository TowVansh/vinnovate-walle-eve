"""Self-checks: python test_walle.py"""
import walle

assert walle.compass(walle.bearing((5, 5), (5, 0))) == "N"
assert walle.compass(walle.bearing((5, 5), (9, 5))) == "E"
assert walle.compass(walle.bearing((5, 5), (2, 8))) == "SW"

# belief stays a probability distribution and moves toward the source
cfg = walle.random_config(3)
a = walle.Agent(cfg)
assert abs(sum(a.belief.values()) - 1) < 1e-9
env = walle.Env(cfg)
env.pos = a.pos = tuple(cfg["eve"]["pos"])
sig = env._signal()
a.update_signal(a.pos, sig["strength"], sig["bearing_deg"])
assert abs(sum(a.belief.values()) - 1) < 1e-9
assert walle.dist(a.top(1)[0][0], cfg["eve"]["pos"]) <= 2

# battery safety: the agent never strands itself, across many random missions
for s in range(50):
    env, agent, trace = walle.run_mission(walle.random_config(s))
    assert env.end_reason != "battery", f"seed {s} ran out of battery"
    assert all(r["obs"]["battery"] >= 0 for r in trace)

# the Bayesian agent beats the reactive baseline
r = walle.bench(40)
assert r["bayes"]["success_rate"] > r["reactive"]["success_rate"], r
print("all checks passed", r["bayes"]["success_rate"], "vs", r["reactive"]["success_rate"])
