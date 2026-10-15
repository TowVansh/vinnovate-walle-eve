# WALL-E and EVE's Signal

An autonomous agent (WALL-E) explores a simulated grid world, reads a noisy beacon, picks up clues, manages a limited battery and time budget, and works out where EVE is and what state she is in. Everything is in one file, `walle.py`, using only the Python standard library.

```
python walle.py run configs/decoy_mission.json   # run one mission: writes the trace, the report and an HTML replay to out/
python walle.py run configs/clean_mission.json
python walle.py gen --seed 42 > configs/my.json  # generate a random mission
python walle.py random                           # 20 brand-new random maps in one replay page, opens in the browser
python walle.py bench --n 200                    # Bayesian agent vs reactive baseline on 200 random missions
python test_walle.py                             # self-checks
```

Each run writes these files to `out/`:
- `<name>_trace.jsonl`: one line per action, with the action, WALL-E's reasoning, the observation, his mode (search or return) and his top 3 location hypotheses.
- `<name>_report.md`: the final mission report, with the outcome, the inferred location and status, confidence, the evidence log, the recommended next step and an ASCII map.
- `<name>_replay.html`: open it in a browser for an animated replay of the path, the belief and the reasoning.

`python walle.py random` writes `out/random_missions.html`: 20 fresh maps (different every time you run it) with a **🎲 Random mission** button and a dropdown to switch between them.

## Environment design

The world is a grid (20x15 by default) with walls, a base where WALL-E can recharge, EVE, optional decoy beacons and clues. `Env` holds the hidden truth. The agent only sees what each action returns.

| Action | Cost | Observation |
|---|---|---|
| `move N/S/E/W` | 1 battery, 1 time | new position (a wall means a bump, no move) |
| `scan` | 2 battery, 1 time | beacon strength + bearing (both noisy), plus any clues within 2.5 cells |
| `contact` | 1 battery, 1 time | whether EVE is in this cell, and her status if she is |
| `recharge` | 10 time | full battery (only at the base) |

The mission ends when EVE is found, the battery hits 0 or the time runs out.

**Sensor:** strength = `strength0 * exp(-d / falloff)` + Gaussian noise, and the bearing gets Gaussian noise in degrees. When a decoy is present, the strongest source wins, so close to the decoy WALL-E hears the spoof instead of EVE.

**Clues** are revealed only by scanning near them. Their content is computed from the hidden truth:
- `tracks`: tread marks pointing toward EVE (45 degree resolution)
- `region`: a beacon log saying which half of the map EVE is in
- `status`: whether EVE is `active` or `dormant`

## Agent

1. **Belief:** a probability over every reachable cell. Each scan does a Bayesian update with a strength likelihood times a von Mises bearing likelihood, mixed with a 15% outlier term so one bad reading cannot wipe out the true cell. Clues multiply in their own likelihoods. A failed contact rules out that cell.
2. **Spoof detection:** if two contacts close together fail even though the signal said EVE was right there, WALL-E marks a decoy zone. He then rebuilds the belief from scratch, replaying all stored observations and dropping the readings the decoy explains: readings taken near it, or pointing at it. This is shown in `configs/decoy_mission.json`.
3. **Planning:** the target cell maximises `P(cell) * exp(-steps / 6)`, considering only cells he can reach and still get home from on the current charge. He scans every 2 to 4 moves while unsure, and stops scanning once he is 60% or more confident, to save battery. He attempts contact once the current cell is likely enough.
4. **Battery safety:** before every action he checks that he can still get home: battery after the action must be at least the BFS distance home plus a margin. If not, he switches to `return` mode, drives home and recharges. If the likely cells are out of range on this charge, he recharges first instead of wandering. Every turn-back and target change is logged as a strategy change.
5. **Report:** the outcome, the top 3 hypotheses, the status estimate, the evidence log and the recommended next step. If he found EVE: tow her home if she is dormant. If not: the exact cell to try next and its battery cost.

## Results (200 random missions, `python walle.py bench --n 200`)

| Agent | Overall | No decoy | With decoy | Mean time when found |
|---|---|---|---|---|
| Bayesian (this) | **86.5%** | **97.0%** | **76.2%** | 78.3 |
| Reactive baseline (follow last bearing) | 59.0% | 72.7% | 45.5% | 93.5 |

The reactive baseline is a classical controller with no belief and no memory. It shows what the belief, the clues and the spoof detection add. The remaining failures are almost all decoy missions where EVE lies roughly in the same direction as the decoy, so the readings dropped as "decoy" also carried EVE's signal.

## Config format (JSON, or YAML if `pyyaml` is installed)

```json
{
  "name": "decoy-mission", "seed": 1129,
  "map": {"width": 20, "height": 15, "walls": [[3, 5], [4, 5]]},
  "start": [1, 3], "base": [1, 3],
  "eve": {"pos": [10, 6], "status": "dormant"},
  "decoys": [{"pos": [4, 2]}],
  "sensor": {"strength0": 100, "falloff": 5, "noise": 3, "bearing_noise_deg": 25,
             "clue_radius": 2.5, "outlier_prob": 0.15, "decoy_strength0": 60},
  "budget": {"battery": 70, "time": 600, "costs": {"move": 1, "scan": 2, "contact": 1, "recharge_time": 10}},
  "clues": [{"pos": [4, 2], "type": "region", "axis": "x"}, {"pos": [9, 9], "type": "tracks"}, {"pos": [2, 12], "type": "status"}],
  "initial_signal": {"strength": 32.3, "bearing_deg": 64.6, "t": 0}
}
```

`eve`, `decoys` and `clues` are the hidden truth: the env uses them and the agent never reads them. The agent knows the map, the base, its own sensor model and the budget.

## Bonus items covered

- Visualization: an ASCII map in the report and an animated HTML replay
- Decoy (spoofed) signals with detection and belief revision
- Noisy sensors
- Comparison against a classical reactive planner
- A generalization benchmark across random missions

## Not done

- Multi-agent coordination
- Comparison against an LLM planner
- Physics simulation (Gazebo/Webots)
