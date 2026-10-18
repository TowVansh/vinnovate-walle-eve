@echo off
rem Live demo: fresh random maps, one full mission with graphs, and the benchmark. Double-click to run.
cd /d "%~dp0"
echo === 20 brand-new random maps (opens in the browser) ===
py walle.py random
echo.
echo === Decoy mission: trace, report, replay and path graph ===
py walle.py run configs\decoy_mission.json
start "" "out\decoy_mission_path.png"
echo.
echo === Benchmark: 200 random missions, Bayesian agent vs reactive baseline (about 18 s) ===
py walle.py bench --n 200
start "" "out\bench.png"
echo.
pause
