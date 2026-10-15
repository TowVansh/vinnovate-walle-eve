#!/usr/bin/env python3
"""WALL-E and EVE's Signal: a simulated world plus an autonomous Bayesian search agent.

Standard library only. Usage:
    python walle.py run configs/sample_mission.json      # one mission: trace + report + replay
    python walle.py gen --seed 7 > configs/random_7.json  # random mission config
    python walle.py bench --n 100                         # Bayesian agent vs reactive baseline
"""
import argparse
import json
import math
import os
import random
import sys
from collections import deque

DIRS = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def bearing(a, b):
    """Compass bearing in degrees from cell a to cell b (0 = north, 90 = east, y grows south)."""
    return math.degrees(math.atan2(b[0] - a[0], -(b[1] - a[1]))) % 360


def compass(deg):
    return COMPASS[int(((deg % 360) + 22.5) // 45) % 8]


def angle_diff(a, b):
    return math.radians((a - b + 180) % 360 - 180)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------- environment

class Env:
    """Holds the hidden truth (EVE, clues, decoys) and answers one observation per action."""

    def __init__(self, cfg):
        self.cfg = cfg
        m = cfg["map"]
        self.W, self.H = m["width"], m["height"]
        self.walls = {tuple(w) for w in m.get("walls", [])}
        self.pos = tuple(cfg["start"])
        self.base = tuple(cfg["base"])
        self.eve = tuple(cfg["eve"]["pos"])
        self.eve_status = cfg["eve"]["status"]
        self.sensor = cfg["sensor"]
        self.decoys = [tuple(d["pos"]) for d in cfg.get("decoys", [])]
        b = cfg["budget"]
        self.max_battery = self.battery = b["battery"]
        self.time_left = b["time"]
        self.costs = b["costs"]
        self.clues = cfg.get("clues", [])
        self.revealed = set()
        self.rng = random.Random(cfg.get("seed", 0) * 7919 + 1)
        self.done, self.found, self.end_reason = False, False, None

    def passable(self, p):
        return 0 <= p[0] < self.W and 0 <= p[1] < self.H and p not in self.walls

    def _spend(self, battery, time):
        self.battery -= battery
        self.time_left -= time

    def _signal(self):
        """Strongest source wins: EVE or a spoofed decoy beacon. Strength and bearing are noisy."""
        s = self.sensor
        readings = []
        for src, power in [(self.eve, s["strength0"])] + [(d, s.get("decoy_strength0", s["strength0"])) for d in self.decoys]:
            r = power * math.exp(-dist(self.pos, src) / s["falloff"]) + self.rng.gauss(0, s["noise"])
            readings.append((r, src))
        r, src = max(readings)
        b = bearing(self.pos, src) + self.rng.gauss(0, s["bearing_noise_deg"]) if src != self.pos else self.rng.uniform(0, 360)
        return {"strength": round(max(r, 0.0), 2), "bearing_deg": round(b % 360, 1)}

    def _clue_view(self, i, c):
        p = tuple(c["pos"])
        view = {"id": i, "pos": list(p), "type": c["type"]}
        if c["type"] == "tracks":  # tread marks point roughly at EVE (45 degree resolution)
            d = round(bearing(p, self.eve) / 45) * 45 % 360
            view.update(bearing_deg=d, text=f"Fresh tread marks heading {compass(d)}")
        elif c["type"] == "region":
            west = self.eve[0] < self.W / 2
            north = self.eve[1] < self.H / 2
            half = c.get("axis", "x")
            view.update(axis=half, value=("west" if west else "east") if half == "x" else ("north" if north else "south"),
                        text=f"Beacon log: EVE last pinged from the {('west' if west else 'east') if half == 'x' else ('north' if north else 'south')} half")
        elif c["type"] == "status":
            view.update(status=self.eve_status, text=f"Plant-pod casing nearby: EVE looks {self.eve_status}")
        return view

    def step(self, action, arg=None):
        if self.done:
            raise RuntimeError("mission already over")
        c = self.costs
        obs = {}
        if action == "move":
            dx, dy = DIRS[arg]
            nxt = (self.pos[0] + dx, self.pos[1] + dy)
            self._spend(c["move"], 1)
            obs["moved"] = self.passable(nxt)
            if obs["moved"]:
                self.pos = nxt
        elif action == "scan":
            self._spend(c["scan"], 1)
            obs["signal"] = self._signal()
            obs["clues"] = []
            for i, cl in enumerate(self.clues):
                if dist(self.pos, cl["pos"]) <= self.sensor["clue_radius"] and i not in self.revealed:
                    self.revealed.add(i)
                    obs["clues"].append(self._clue_view(i, cl))
        elif action == "recharge":
            if self.pos == self.base:
                self.battery = self.max_battery
                self._spend(0, c["recharge_time"])
            else:
                obs["error"] = "not at base"
                self._spend(0, 1)
        elif action == "contact":
            self._spend(c["contact"], 1)
            self.found = obs["contact"] = self.pos == self.eve
            if self.found:
                obs["eve_status"] = self.eve_status
                self.done, self.end_reason = True, "found"
        else:
            raise ValueError(action)
        if not self.done and self.battery <= 0:
            self.done, self.end_reason = True, "battery"
        elif not self.done and self.time_left <= 0:
            self.done, self.end_reason = True, "time"
        obs.update(pos=list(self.pos), battery=self.battery, time_left=self.time_left)
        return obs


# ---------------------------------------------------------------- agent

class Agent:
    """Bayesian grid belief over EVE's cell, utility-driven target choice, battery-safe returns."""

    name = "bayes"

    def __init__(self, cfg):
        m = cfg["map"]
        self.W, self.H = m["width"], m["height"]
        self.walls = {tuple(w) for w in m.get("walls", [])}
        self.base = tuple(cfg["base"])
        self.pos = tuple(cfg["start"])
        self.sensor = cfg["sensor"]  # WALL-E knows his own sensor model, not where the sources are
        b = cfg["budget"]
        self.max_battery = self.battery = b["battery"]
        self.costs = b["costs"]
        self.cells = [(x, y) for y in range(self.H) for x in range(self.W) if (x, y) not in self.walls]
        self._dist_cache = {}
        home = self.bfs(self.base)
        self.cells = [c for c in self.cells if c in home]  # unreachable cells cannot hold EVE for us
        self.belief = {c: 1 / len(self.cells) for c in self.cells}
        self.status = {"active": 0.5, "dormant": 0.5}
        self.evidence = []
        self.moves_since_scan = 99
        self.target = None
        self.mode = "search"
        self.spoof_flags = 0
        self.scans, self.clue_log, self.ruled_out, self.decoy_zones = [], [], set(), []
        self.strong_misses = []  # failed contacts where the signal claimed EVE was right here
        init = cfg.get("initial_signal")
        if init:
            self.scans.append((self.pos, init["strength"], init["bearing_deg"]))
            self.update_signal(self.pos, init["strength"], init["bearing_deg"])
            self.evidence.append(f"t=0 initial reading: strength {init['strength']}, bearing {init['bearing_deg']} deg ({compass(init['bearing_deg'])})")

    def from_decoy(self, pos, bear):
        """A reading taken near a known decoy, or pointing at one, carries no information about EVE."""
        for z in self.decoy_zones:
            if dist(pos, z) <= 3 or abs(angle_diff(bear, bearing(pos, z))) < math.radians(30):
                return True
        return False

    def rebuild_belief(self):
        """Belief revision: replay every observation from a flat prior, dropping readings explained by decoys."""
        self.belief = {c: 1.0 for c in self.cells}
        self._normalize()
        for pos, s, b in self.scans:
            if not self.from_decoy(pos, b):
                self.update_signal(pos, s, b)
        for cl in self.clue_log:
            self.update_clue(cl)
        for c in self.belief:
            if c in self.ruled_out:
                self.belief[c] = 0.0
            elif any(dist(c, z) <= 2 for z in self.decoy_zones):
                self.belief[c] *= 0.2  # already searched around the decoy
        self._normalize()

    # --- geometry
    def bfs(self, src):
        if src in self._dist_cache:
            return self._dist_cache[src]
        d = {src: 0}
        q = deque([src])
        while q:
            p = q.popleft()
            for dx, dy in DIRS.values():
                n = (p[0] + dx, p[1] + dy)
                if 0 <= n[0] < self.W and 0 <= n[1] < self.H and n not in self.walls and n not in d:
                    d[n] = d[p] + 1
                    q.append(n)
        self._dist_cache[src] = d
        return d

    def step_toward(self, goal):
        d = self.bfs(goal)
        best = min(DIRS.items(), key=lambda kv: d.get((self.pos[0] + kv[1][0], self.pos[1] + kv[1][1]), 1e9))
        return best[0]

    # --- belief updates
    def _normalize(self):
        z = sum(self.belief.values()) or 1.0
        for c in self.belief:
            self.belief[c] /= z

    def update_signal(self, pos, strength, bear):
        s = self.sensor
        kappa = 1 / math.radians(s["bearing_noise_deg"]) ** 2
        eps = s.get("outlier_prob", 0.15)  # mixture keeps one spoofed reading from zeroing the true cell
        prior_fit = 0.0
        new = {}
        for c, p in self.belief.items():
            d = dist(pos, c)
            exp_s = s["strength0"] * math.exp(-d / s["falloff"])
            ls = math.exp(-((strength - exp_s) ** 2) / (2 * s["noise"] ** 2))
            lb = 1.0 if d == 0 else math.exp(kappa * (math.cos(angle_diff(bear, bearing(pos, c))) - 1))
            lik = (1 - eps) * ls * lb + eps
            prior_fit += p * ls * lb
            new[c] = p * lik
        self.belief = new
        self._normalize()
        return prior_fit

    def update_clue(self, clue):
        t = clue["type"]
        if t == "tracks":
            p = tuple(clue["pos"])
            kappa = 1 / math.radians(35) ** 2
            for c in self.belief:
                if c != p:
                    self.belief[c] *= 0.1 + math.exp(kappa * (math.cos(angle_diff(clue["bearing_deg"], bearing(p, c))) - 1))
        elif t == "region":
            for c in self.belief:
                inside = {"west": c[0] < self.W / 2, "east": c[0] >= self.W / 2,
                          "north": c[1] < self.H / 2, "south": c[1] >= self.H / 2}[clue["value"]]
                self.belief[c] *= 0.95 if inside else 0.05
        elif t == "status":
            other = "active" if clue["status"] == "dormant" else "dormant"
            self.status = {clue["status"]: 0.9, other: 0.1}
        self._normalize()

    def top(self, k=3):
        return sorted(self.belief.items(), key=lambda kv: -kv[1])[:k]

    # --- decision
    def home_cost(self, cell):
        return self.bfs(self.base).get(cell, 1e9) * self.costs["move"]

    def decide(self):
        """Returns (action, arg, reason)."""
        c = self.costs
        margin = c["scan"] + c["contact"]
        best_cell, best_p = self.top(1)[0]

        if self.pos == self.base and self.battery < self.max_battery * 0.7:
            self.mode = "search"
            return "recharge", None, f"at base with {self.battery}/{self.max_battery} battery, topping up before next leg"

        if self.mode == "return" or self.battery - c["move"] - c["scan"] < self.home_cost(self.pos) + margin:
            if self.mode != "return":
                self.mode = "return"
                self.evidence.append(f"turned back at {list(self.pos)} with battery {self.battery} (home costs {self.home_cost(self.pos)})")
            return "move", self.step_toward(self.base), f"returning to recharge: battery {self.battery}, home costs {self.home_cost(self.pos)}"

        if self.belief.get(self.pos, 0) >= 0.35 or (self.pos == best_cell and best_p > 0.12):
            return "contact", None, f"P(EVE here)={self.belief.get(self.pos, 0):.2f}, attempting contact"

        scan_every = 2 if best_p > 0.15 else 4
        if best_p < 0.6 and self.moves_since_scan >= scan_every:  # confident: save battery, just drive there
            return "scan", None, ("first scan of the mission" if self.moves_since_scan >= 99 else f"{self.moves_since_scan} moves since last scan") + f" (scan every {scan_every} at max belief {best_p:.2f})"

        # utility = belief mass near a cell, discounted by travel; only cells we can reach and still get home from
        here = self.bfs(self.pos)
        reach = self.battery - margin
        best, best_u = None, -1
        for cell, p in self.belief.items():
            d = here.get(cell)
            if d is None or d * c["move"] + self.home_cost(cell) > reach:
                continue
            u = p * math.exp(-d / 6)
            if u > best_u:
                best, best_u = cell, u
        if best is not None and self.belief[best] < 0.25 * best_p and self.pos != self.base:
            best = None  # the likely cells are out of range on this charge: recharge instead of wandering
        if best is None or best == self.pos:
            if best is None:
                self.mode = "return"
                return "move", self.step_toward(self.base), "no belief mass reachable on this charge, heading home"
            return "scan", None, "best target is current cell, scanning to sharpen belief"
        if best != self.target:
            if self.target is not None:
                self.evidence.append(f"strategy change: target {list(self.target)} -> {list(best)} (P={self.belief[best]:.2f})")
            self.target = best
        return "move", self.step_toward(best), f"heading to {list(best)} (P={self.belief[best]:.2f}, {here[best]} steps)"

    def observe(self, action, obs):
        self.pos = tuple(obs["pos"])
        self.battery = obs["battery"]
        if action == "move":
            self.moves_since_scan += 1
        elif action == "scan":
            self.moves_since_scan = 0
            sig = obs["signal"]
            self.scans.append((self.pos, sig["strength"], sig["bearing_deg"]))
            if self.from_decoy(self.pos, sig["bearing_deg"]):
                pass  # explained by a known decoy, ignore
            elif self.update_signal(self.pos, sig["strength"], sig["bearing_deg"]) < 1e-4:
                self.spoof_flags += 1
                self.evidence.append(f"reading at {list(self.pos)} (strength {sig['strength']}, {compass(sig['bearing_deg'])}) contradicts belief: possible spoof, down-weighted")
            for cl in obs["clues"]:
                self.clue_log.append(cl)
                self.update_clue(cl)
                self.evidence.append(f"clue at {cl['pos']}: {cl['text']}")
        elif action == "contact" and not obs["contact"]:
            self.ruled_out.add(self.pos)
            self.belief[self.pos] = 0.0
            self._normalize()
            self.evidence.append(f"contact failed at {list(self.pos)}, cell ruled out")
            recent = [s for p, s, _ in self.scans[-3:] if dist(p, self.pos) <= 2]
            if recent and max(recent) > self.sensor["strength0"] * 0.35:
                self.strong_misses.append(self.pos)
            near = [p for p in self.strong_misses if dist(p, self.pos) <= 3]
            if len(near) >= 2 and not any(dist(z, self.pos) <= 4 for z in self.decoy_zones):
                z = (round(sum(p[0] for p in near) / len(near)), round(sum(p[1] for p in near) / len(near)))
                self.decoy_zones.append(z)
                self.spoof_flags += 1
                self.rebuild_belief()
                (bc, bp), = self.top(1)
                self.evidence.append(f"SPOOF DETECTED: strong signal but no EVE around {list(z)} after {len(near)} contacts; "
                                     f"marked as decoy, belief rebuilt without its readings (new best {list(bc)}, P={bp:.2f})")
        elif action == "contact":
            self.belief = {c: float(c == self.pos) for c in self.belief}
            self.status = {obs["eve_status"]: 1.0}
            self.evidence.append(f"contact made with EVE at {list(self.pos)}, status {obs['eve_status']}")


class ReactiveAgent(Agent):
    """Classical baseline: follow the latest bearing, contact when the signal is strong. No belief, no memory."""

    name = "reactive"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.last = cfg.get("initial_signal")
        self.tried = set()

    def decide(self):
        c = self.costs
        if self.pos == self.base and self.battery < self.max_battery * 0.7:
            return "recharge", None, "recharge"
        if self.mode == "return" or self.battery - c["move"] - c["scan"] < self.home_cost(self.pos) + c["scan"] + c["contact"]:
            self.mode = "return"
            if self.pos == self.base:
                self.mode = "search"
                return "recharge", None, "recharge"
            return "move", self.step_toward(self.base), "battery low"
        if self.last and self.last["strength"] > self.sensor["strength0"] * 0.8 and self.pos not in self.tried:
            self.tried.add(self.pos)
            return "contact", None, "strong signal"
        if self.moves_since_scan >= 2:
            return "scan", None, "periodic scan"
        b = self.last["bearing_deg"] if self.last else 0
        options = sorted(DIRS, key=lambda k: abs(angle_diff(b, bearing((0, 0), DIRS[k]))))
        for k in options:
            n = (self.pos[0] + DIRS[k][0], self.pos[1] + DIRS[k][1])
            if n in self.bfs(self.base):
                return "move", k, f"follow bearing {b:.0f}"
        return "scan", None, "stuck"

    def observe(self, action, obs):
        super().observe(action, obs)
        if action == "scan":
            self.last = obs["signal"]
        if self.mode == "return" and self.pos == self.base:
            self.mode = "search"


# ---------------------------------------------------------------- mission runner

def run_mission(cfg, agent_cls=Agent, trace_path=None, max_steps=5000):
    env, agent = Env(cfg), agent_cls(cfg)
    trace = []
    for t in range(max_steps):
        if env.done:
            break
        action, arg, reason = agent.decide()
        obs = env.step(action, arg)
        agent.observe(action, obs)
        if action == "recharge" and agent.mode == "return":
            agent.mode = "search"
        trace.append({"t": t, "action": action, "arg": arg, "reason": reason, "obs": obs, "mode": agent.mode,
                      "top_belief": [[list(c), round(p, 3)] for c, p in agent.top(3)]})
    if not env.done:
        env.end_reason = "step_limit"
    if trace_path:
        with open(trace_path, "w") as f:
            for row in trace:
                f.write(json.dumps(row) + "\n")
    return env, agent, trace


def make_report(cfg, env, agent, trace):
    (cell, p), = agent.top(1)
    status = max(agent.status, key=agent.status.get)
    budget = cfg["budget"]
    recharges = sum(1 for r in trace if r["action"] == "recharge")
    lines = [f"# Mission report: {cfg.get('name', 'mission')}", "",
             f"- Outcome: **{'EVE FOUND' if env.found else 'EVE NOT FOUND'}** (ended by: {env.end_reason})",
             f"- Steps: {len(trace)} | time used: {budget['time'] - env.time_left}/{budget['time']} | battery left: {env.battery}/{budget['battery']} | recharges: {recharges}",
             f"- Inferred EVE location: {list(cell)} with confidence {p:.0%}",
             f"- Inferred EVE status: {status} ({agent.status[status]:.0%})",
             f"- Spoofed/inconsistent readings flagged: {agent.spoof_flags}", "",
             "## Top hypotheses", ""]
    lines += [f"{i + 1}. {list(c)}: {q:.1%}" for i, (c, q) in enumerate(agent.top(3))]
    lines += ["", "## Evidence log", ""] + [f"- {e}" for e in agent.evidence]
    if env.found:
        nxt = f"Dock with EVE at {list(cell)} and escort her back to base {list(agent.base)}" + (" (she is dormant: tow, do not wait for a response)" if status == "dormant" else "")
    else:
        need = agent.bfs(agent.base).get(cell, 0) * cfg["budget"]["costs"]["move"]
        nxt = f"Recharge, then go straight to {list(cell)} ({need} battery each way) and attempt contact; next best: {[list(c) for c, _ in agent.top(3)[1:]]}"
    lines += ["", "## Recommended next step", "", nxt, "", "## Map (post-mission debrief, truth revealed)", "", "```", render_map(cfg, env, agent, trace), "```"]
    return "\n".join(lines) + "\n"


def render_map(cfg, env, agent, trace):
    """ASCII: # wall, B base, E true EVE, D decoy, ? clue, * path, belief shading . : + @ elsewhere."""
    path = {tuple(r["obs"]["pos"]) for r in trace}
    clues = {tuple(c["pos"]) for c in cfg.get("clues", [])}
    pmax = max(agent.belief.values()) or 1
    rows = []
    for y in range(env.H):
        row = ""
        for x in range(env.W):
            p = (x, y)
            if p in env.walls:
                ch = "#"
            elif p == env.eve:
                ch = "E"
            elif p == env.base:
                ch = "B"
            elif p in env.decoys:
                ch = "D"
            elif p in clues:
                ch = "?"
            elif p in path:
                ch = "*"
            else:
                ch = " .:+@"[min(4, int(4 * agent.belief.get(p, 0) / pmax + 0.5))] if agent.belief.get(p, 0) > 0.01 * pmax else " "
            row += ch
        rows.append(row)
    border = "+" + "-" * env.W + "+"
    return "\n".join([border] + ["|" + r + "|" for r in rows] + [border])


def write_replay(cfg, trace, path, more=()):
    """Self-contained HTML replay (path, belief top-3, battery, reasoning). `more` adds extra (cfg, trace, found) missions."""
    found = trace and trace[-1]["action"] == "contact" and trace[-1]["obs"].get("contact")
    missions = [{"cfg": cfg, "trace": trace, "found": bool(found)}] + [{"cfg": c, "trace": t, "found": f} for c, t, f in more]
    html = REPLAY_HTML.replace("__MISSIONS__", json.dumps(missions, separators=(",", ":")))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------- random missions

def random_config(seed, w=20, h=15):
    rng = random.Random(seed)
    while True:
        walls = {(x, y) for x in range(w) for y in range(h) if rng.random() < 0.14}
        base = (rng.randrange(3), rng.randrange(h))
        walls.discard(base)
        a = Agent({"map": {"width": w, "height": h, "walls": list(walls)}, "base": list(base), "start": list(base),
                   "sensor": {"bearing_noise_deg": 25}, "budget": {"battery": 1, "costs": {}}})
        reach = a.bfs(base)
        far = [c for c, d in reach.items() if d >= 12]
        if len(far) > 20:
            break
    eve = rng.choice(far)
    free = [c for c in reach if c not in (eve, base)]
    clue_cells = rng.sample(free, 5)
    sensor = {"strength0": 100, "falloff": 5, "noise": 3, "bearing_noise_deg": 25, "clue_radius": 2.5, "outlier_prob": 0.15}
    cfg = {
        "name": f"random-{seed}", "seed": seed,
        "map": {"width": w, "height": h, "walls": sorted(map(list, walls))},
        "start": list(base), "base": list(base),
        "eve": {"pos": list(eve), "status": rng.choice(["active", "dormant"])},
        "sensor": sensor,
        "budget": {"battery": 70, "time": 600, "costs": {"move": 1, "scan": 2, "contact": 1, "recharge_time": 10}},
        "clues": [{"pos": list(clue_cells[0]), "type": "region", "axis": "x"},
                  {"pos": list(clue_cells[1]), "type": "region", "axis": "y"},
                  {"pos": list(clue_cells[2]), "type": "status"},
                  {"pos": list(clue_cells[3]), "type": "tracks"},
                  {"pos": list(clue_cells[4]), "type": "tracks"}],
    }
    if rng.random() < 0.5:
        cfg["decoys"] = [{"pos": list(rng.choice(free))}]
        sensor["decoy_strength0"] = 60
    # initial reading taken at the start cell, like the scan action would
    e = Env(cfg)
    cfg["initial_signal"] = dict(e._signal(), t=0)
    return cfg


def bench(n, seed0=1000):
    out = {}
    for cls in (Agent, ReactiveAgent):
        found, time_used, by_kind = 0, [], {"clean": [0, 0], "decoy": [0, 0]}
        for s in range(seed0, seed0 + n):
            cfg = random_config(s)
            env, agent, trace = run_mission(cfg, cls)
            found += env.found
            k = by_kind["decoy" if cfg.get("decoys") else "clean"]
            k[0] += env.found
            k[1] += 1
            if env.found:
                time_used.append(cfg["budget"]["time"] - env.time_left)
        out[cls.name] = {"missions": n, "success_rate": round(found / n, 3),
                         "success_no_decoy": round(by_kind["clean"][0] / max(1, by_kind["clean"][1]), 3),
                         "success_with_decoy": round(by_kind["decoy"][0] / max(1, by_kind["decoy"][1]), 3),
                         "mean_time_when_found": round(sum(time_used) / len(time_used), 1) if time_used else None}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("config")
    r.add_argument("--out", default="out")
    r.add_argument("--agent", choices=["bayes", "reactive"], default="bayes")
    g = sub.add_parser("gen")
    g.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("bench")
    b.add_argument("--n", type=int, default=100)
    rnd = sub.add_parser("random", help="run N fresh random missions into one replay page with a Random mission button")
    rnd.add_argument("--n", type=int, default=20)
    rnd.add_argument("--out", default="out")
    rnd.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    if a.cmd == "gen":
        print(json.dumps(random_config(a.seed), indent=1))
    elif a.cmd == "bench":
        print(json.dumps(bench(a.n), indent=2))
    elif a.cmd == "random":
        os.makedirs(a.out, exist_ok=True)
        runs = []
        for seed in random.sample(range(1, 10**6), a.n):  # new maps every time you run it
            cfg = random_config(seed)
            env, agent, trace = run_mission(cfg)
            runs.append((cfg, trace, env.found))
        path = os.path.join(a.out, "random_missions.html")
        write_replay(runs[0][0], runs[0][1], path, more=runs[1:])
        print(f"{sum(f for _, _, f in runs)}/{a.n} missions found EVE. Wrote {path}")
        if not a.no_open:
            import webbrowser
            webbrowser.open("file://" + os.path.abspath(path))
    else:
        with open(a.config) as f:
            if a.config.endswith((".yaml", ".yml")):
                import yaml  # optional: pip install pyyaml
                cfg = yaml.safe_load(f)
            else:
                cfg = json.load(f)
        os.makedirs(a.out, exist_ok=True)
        stem = os.path.join(a.out, os.path.splitext(os.path.basename(a.config))[0])
        env, agent, trace = run_mission(cfg, Agent if a.agent == "bayes" else ReactiveAgent, trace_path=stem + "_trace.jsonl")
        report = make_report(cfg, env, agent, trace)
        with open(stem + "_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        write_replay(cfg, trace, stem + "_replay.html")
        sys.stdout.write(report)
        print(f"\nwrote {stem}_trace.jsonl, {stem}_report.md, {stem}_replay.html")


REPLAY_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>WALL-E mission replay</title>
<style>body{font:14px system-ui;background:#1b1a17;color:#eee;margin:16px}canvas{background:#2a2722;image-rendering:pixelated;max-width:100%}
#log{font-family:monospace;white-space:pre-wrap;background:#111;padding:8px;height:120px;overflow:auto}button,input,select{font-size:14px}
button{background:#facc15;border:0;border-radius:6px;padding:6px 12px;font-weight:600;cursor:pointer}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0}
#outcome{font-weight:600}</style></head><body>
<h2>WALL-E mission replay</h2>
<div class="row"><button id="rand">🎲 Random mission</button><select id="pick"></select><span id="outcome"></span></div>
<div class="row"><button id="play">▶ Play</button> <input id="t" type="range" min="0" value="0" style="flex:1;min-width:200px"> <span id="info"></span></div>
<canvas id="c"></canvas><div id="log"></div>
<p>Yellow = WALL-E, trail = path, green = base, pink = EVE (truth), red = decoy, blue = clue, orange squares = agent's top-3 belief.</p>
<script>
const missions=__MISSIONS__, S=28;
let cfg, trace, cur=-1, timer=null;
const cv=document.getElementById('c'), g=cv.getContext('2d'), t=document.getElementById('t'), pick=document.getElementById('pick');
missions.forEach((m,i)=>pick.insertAdjacentHTML('beforeend',`<option value="${i}">${m.cfg.name} (${m.found?'found':'not found'})</option>`));
function load(i){
 cur=i; pick.value=i; cfg=missions[i].cfg; trace=missions[i].trace;
 cv.width=cfg.map.width*S; cv.height=cfg.map.height*S; t.max=trace.length-1; t.value=0;
 const m=missions[i];
 document.getElementById('outcome').textContent=`${m.found?'✅ EVE found':'❌ EVE not found'} in ${trace.length} steps${cfg.decoys?' · decoy beacon on this map':''}`;
 stop(); draw(0);
}
function cell(p,col,pad=2){g.fillStyle=col;g.fillRect(p[0]*S+pad,p[1]*S+pad,S-2*pad,S-2*pad)}
function draw(i){g.clearRect(0,0,cv.width,cv.height);
 cfg.map.walls.forEach(w=>cell(w,'#555',0));
 (cfg.clues||[]).forEach(c=>cell(c.pos,'#3b82f6',8));
 (cfg.decoys||[]).forEach(d=>cell(d.pos,'#ef4444',6));
 cell(cfg.base,'#22c55e',3); cell(cfg.eve.pos,'#ec4899',4);
 const r=trace[i]; r.top_belief.forEach(([p,q],k)=>{g.strokeStyle='rgba(249,115,22,'+(1-k*0.3)+')';g.lineWidth=3;g.strokeRect(p[0]*S+1,p[1]*S+1,S-2,S-2)});
 g.fillStyle='rgba(250,204,21,.35)'; for(let k=0;k<=i;k++){const p=trace[k].obs.pos;g.fillRect(p[0]*S+10,p[1]*S+10,S-20,S-20)}
 cell(r.obs.pos,'#facc15',5);
 document.getElementById('info').textContent=`step ${i} | battery ${r.obs.battery} | time left ${r.obs.time_left} | mode ${r.mode}`;
 document.getElementById('log').textContent=trace.slice(Math.max(0,i-6),i+1).map(x=>`${x.t} ${x.action}${x.arg?' '+x.arg:''}: ${x.reason}`).join('\n');}
function stop(){clearInterval(timer);timer=null;document.getElementById('play').textContent='▶ Play'}
t.oninput=()=>draw(+t.value);
document.getElementById('play').onclick=()=>{if(timer)return stop();
 if(+t.value>=trace.length-1)t.value=0;
 document.getElementById('play').textContent='⏸ Pause';
 timer=setInterval(()=>{if(+t.value>=trace.length-1)return stop();t.value=+t.value+1;draw(+t.value)},60)};
pick.onchange=()=>load(+pick.value);
document.getElementById('rand').onclick=()=>{let i;do i=Math.floor(Math.random()*missions.length);while(missions.length>1&&i===cur);load(i);document.getElementById('play').click()};
load(0);
</script></body></html>"""

if __name__ == "__main__":
    main()
