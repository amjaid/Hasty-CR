# Hasty-CR Training & Live Runbook

Last verified: **2026-10-17**  
Repo root: `C:\Users\jaid2\OneDrive\Documents\GitHub\Hasty-CR`  
Live device: **MuMu instance 0, `127.0.0.1:16384`** (confirmed to have Clash Royale installed)

This is the practical "how do I train, evaluate, and play again" document. It
matches the commands and file paths in the current checkout. For the design
rationale behind the simulator and policy, read `README.md` and
`docs/SIM_MECHANICS.md`.

---

## 0. Quick reference

```powershell
# Play with the current best checkpoint (friendly: you start the battle)
.\run.ps1 -Brain rl -Checkpoint "tmp\rl\ppo_best.pt" -NoQueue

# Play ladder matches (bot presses Battle itself)
.\run.ps1 -Brain rl -Checkpoint "tmp\rl\ppo_best.pt" -Matches 5

# Evaluate before trusting a checkpoint
.venvs\buildabot\Scripts\python.exe scripts\evaluate_pilot.py `
    --ckpt tmp\rl\ppo_best.pt --episodes 60

# Stop everything the launcher started
.\run.ps1 -Stop
```

Back up any checkpoint you care about. `tmp\`, `checkpoints\`, and `*.pt` are
gitignored, so a fresh clone will not contain your trained weights.

---

## 1. Environment

### 1.1 The interpreter `run.ps1` uses

`run.ps1` hardcodes:

```text
.venvs\buildabot\Scripts\python.exe
```

That virtualenv is deliberately **not** the system Python. It contains:

- Python 3.12 (the project requires `>=3.12,<3.13`)
- `clashroyalebuildabot` and its ONNX models
- HastyCR's `[live,dev]` extras: torch, torchvision, Ultralytics, OpenCV,
  Pillow, psutil, PyQt6, av, pytest
- NumPy `2.4.6` (pinned by upstream BuildABot)

### 1.2 Fresh-machine setup

From the repo root in PowerShell:

```powershell
# One-time: install uv if it is not already available
pip install uv

# Clone the upstream perception package at the pinned commit
git clone https://github.com/Pbatch/ClashRoyaleBuildABot.git vendor\ClashRoyaleBuildABot
git -C vendor\ClashRoyaleBuildABot checkout f89ecbf309e698cb9fccdfb64a6ff2a930e6541e

# Create the live venv
uv venv .venvs\buildabot --python 3.12

# Install upstream BuildABot (CPU ONNX runtime)
uv pip install --python .venvs\buildabot\Scripts\python.exe `
    -e "vendor/ClashRoyaleBuildABot[cpu]"

# Install HastyCR plus the live/test dependencies
uv pip install --python .venvs\buildabot\Scripts\python.exe -e ".[live,dev]"
```

`scripts\bootstrap.ps1` does the same setup plus the other vendored research
repos. It creates three venvs and clones five repositories, so it is slower than
the five commands above. For live play, the `buildabot` venv is the only one
that matters.

### 1.3 Verify the environment

```powershell
.venvs\buildabot\Scripts\python.exe -c `
  "import numpy, torch, cv2, ultralytics, clashroyalebuildabot; print(numpy.__version__, torch.__version__, torch.cuda.is_available())"
```

Expected on this machine:

```text
2.4.6 2.14.1+cpu False
```

The CPU torch build is fine for live RL inference: the policy network is tiny.
It is also fine for small simulator runs, but PPO training is much faster with a
CUDA build. To use the RTX 4050, reinstall torch from the CUDA wheel selected on
<https://pytorch.org/get-started/locally/> into `.venvs\buildabot`, then confirm
`torch.cuda.is_available()` returns `True`. `sim.train_ppo` and `sim.clone`
select CUDA automatically when it is available.

### 1.4 Run the tests

```powershell
.venvs\buildabot\Scripts\python.exe -m pytest tests -q
```

For just the RL/vision path:

```powershell
.venvs\buildabot\Scripts\python.exe -m pytest `
    tests\test_rl_policy.py tests\test_emulator_capture.py tests\test_brain.py -q
```

---

## 2. Emulator and device identity

### 2.1 Current device

MuMu instance **0** is named `Android Device` and answers on port **16384**.
The repository's defaults were changed from `16480` to `16384` in `run.ps1`,
`scripts\cr_bot.py`, `scripts\emulator.py`, `scripts\supervisor.py`, and the
other live/capture scripts.

Check it yourself:

```powershell
$adb = "C:\Program Files\Netease\MuMuPlayer\nx_device\15.0\shell\adb.exe"
$mgr = "C:\Program Files\Netease\MuMuPlayer\nx_main\MuMuManager.exe"

& $mgr info -v all
& $mgr adb -v 0
& $adb connect 127.0.0.1:16384
& $adb devices
& $adb -s 127.0.0.1:16384 shell pm list packages |
    Select-String "supercell"
```

You must see `com.supercell.clashroyale`. `scripts\emulator.py` refuses to run
if the serial resolves to `com.supercell.clashofclans`; that guard exists so a
wrong port cannot send taps into someone else's game.

### 2.2 Changing the port on another machine

Ports are per-machine. If MuMu reports a different port:

1. Get the real port from `MuMuManager.exe adb -v <index>`.
2. Set it once via the environment:

   ```powershell
   $env:HASTYCR_SERIAL = "127.0.0.1:16384"
   $env:HASTYCR_INSTANCE = "Android Device"
   ```

   `cr_bot.py`, `emulator.py`, and the studio read these overrides.

3. Or change `$serial` in `run.ps1` and the defaults in `scripts\emulator.py`
   and `scripts\cr_bot.py` if you want the change to be permanent.

The legacy ports `7555` and `5555` are not used. Do not point the bot at an
instance unless `pm list packages` shows Clash Royale on it.

---

## 3. Training

### 3.1 What the policy sees and chooses

- Observation: `8 × 32 × 18` float planes plus `47` scalars.
- Action space: `2,321` actions = `1` hold + `4` hand slots × `576` tiles.
- Illegal actions are masked before the softmax.
- Checkpoints are PyTorch dictionaries with at least `state_dict` and `step`.
  `sim.train_ppo` also saves the optimiser state.

### 3.2 Recommended first run: clone, then PPO

PPO from random weights spends a very long time discovering that it should
attack. Start from a behaviour clone of the hand-written brain.

```powershell
# Step 1: behaviour clone (writes tmp\rl\clone_v2.pt)
.venvs\buildabot\Scripts\python.exe -m sim.clone `
    --episodes 4000 --epochs 8 --name clone_v2

# Step 2: PPO from the clone (writes tmp\rl\ppo_v2_best.pt / _last.pt)
.venvs\buildabot\Scripts\python.exe -m sim.train_ppo `
    --envs 8 --steps 2000000 `
    --init tmp\rl\clone_v2.pt `
    --opponent meta --brain-share 0.5 `
    --value-warmup 300000 `
    --name ppo_v2
```

Use a **new `--name`** for every run. Reusing `--name ppo` overwrites the
`ppo_best.pt` you are currently playing with.

`sim.clone` defaults to `--episodes 400`; `4000` is the README's larger,
better starting point. The clone trains only the action head, which is why PPO
needs `--value-warmup` (it fits the randomly initialised critic before letting
the policy move).

### 3.3 Flags that matter

| flag | what it does |
|---|---|
| `--init <ckpt>` | Start from weights only. Use this to begin PPO from a clone. |
| `--resume <ckpt>` | Continue a PPO run, including optimiser state and step count. |
| `--value-warmup 300000` | Freeze the policy while the critic catches up. Essential with `--init`. |
| `--target-kl 0.02` | Stop an epoch once the policy has moved this far. Prevents the clone from being destroyed. |
| `--opponent meta\|brain\|simple\|mirror` | Training opponent. `meta` samples real ladder decks. |
| `--brain-share 0.5` | Of non-self-play episodes, the fraction against the rule engine. Keep this mixed. |
| `--scripted-alt meta\|mirror\|simple` | The other half of the scripted diet. |
| `--league N` | Keep `N` past selves as self-play opponents. |
| `--scripted-share 0.4` | Fraction of league episodes still played against scripted decks. Never set to `0`. |
| `--eval-opponent brain,meta` | Select checkpoints on the **worst** of several opponents. |
| `--entropy` / `--entropy-final` / `--entropy-anneal` | Explore-then-exploit schedule. |
| `--crown` / `--win` / `--chip` / `--elixir` | Reward weights. |
| `--envs` / `--rollout` | Parallel environments and rollout length. |

Defaults worth knowing: `--steps 2000000`, `--envs 6`, `--rollout 128`,
`--lr 3e-4`, `--eval-every 150000`, `--eval-episodes 12`, `--name ppo`.

### 3.4 Outputs and logs

| path | contents |
|---|---|
| `tmp\rl\<name>.pt` | Behaviour clone. |
| `tmp\rl\<name>_best.pt` | Best PPO checkpoint by training eval score. |
| `tmp\rl\<name>_best.json` | The eval that selected it. |
| `tmp\rl\<name>_last.pt` | Latest checkpoint, including optimiser state. |
| `tmp\rl\<name>_league\` | Self-play snapshots when `--league` is used. |
| `tmp\live\rl_train.log` | Appended training log. |
| `checkpoints\<name>\best.pt` | Promoted checkpoint from `rl_supervisor.py`. |
| `checkpoints\<name>\supervisor.log` | Supervisor decisions and final audit. |

`tmp\` is scratch and is overwritten by future runs. A checkpoint meant for
real play belongs under `checkpoints\<name>\` with a manifest of what it scored.

### 3.5 Unattended / overnight training

`scripts\rl_supervisor.py` watches behaviour, not just loss. It stops a run that
collapses, rolls back to the best checkpoint, and restarts with tamer settings.

```powershell
.venvs\buildabot\Scripts\python.exe scripts\rl_supervisor.py `
    --name night12 --hours 6 --envs 8 `
    --init tmp\rl\clone_v2.pt
```

It writes `checkpoints\night12\best.pt` and `supervisor.log`. It does **not**
promote a checkpoint to live play automatically; run the held-out evaluation in
section 4 first. Use a unique `--name` per supervisor so two runs cannot
overwrite each other's promoted checkpoint.

### 3.6 Other trainer

There is also a token/transformer imitation trainer (`sim\train_tokens.py`) used
for the replay-data experiments:

```powershell
.venvs\buildabot\Scripts\python.exe -m sim.train_tokens `
    --name rs_huge --epochs 3 --wait-keep 0.25 `
    --data data\rs_rows --d 384 --layers 8 --heads 8
```

This requires the prepared `data\rs_rows` corpus and is separate from the live
PPO checkpoint format. Use `sim.clone` + `sim.train_ppo` for the live RL path.

---

## 4. Evaluation before playing

Training's `EVAL` line is a smoke test, not proof. Evaluate on fixed held-out
seeds and more than one opponent.

```powershell
# 60 games against the default opponents: brain, meta, simple
.venvs\buildabot\Scripts\python.exe scripts\evaluate_pilot.py `
    --ckpt tmp\rl\ppo_v2_best.pt --episodes 60

# Larger final check on the fresh held-out seed
.venvs\buildabot\Scripts\python.exe scripts\evaluate_pilot.py `
    --ckpt tmp\rl\ppo_v2_best.pt --episodes 200 --seed 9000

# Evaluate every tmp\rl\*_best.pt file
.venvs\buildabot\Scripts\python.exe scripts\evaluate_pilot.py `
    --ckpt tmp\rl\ppo_v2_best.pt --all

# Compare two checkpoints on the same deck and seeds
.venvs\buildabot\Scripts\python.exe scripts\head_to_head.py `
    --a tmp\rl\ppo_v2_best.pt --b tmp\rl\ppo_best.pt --games 100

# Look for degenerate one-card / one-tile strategies
.venvs\buildabot\Scripts\python.exe scripts\exploit_probe.py `
    --ckpt tmp\rl\ppo_v2_best.pt --episodes 60
```

Reports land in `reports\eval_<checkpoint>.json` and
`reports\exploit_<checkpoint>.json` unless `--out` is passed.

What to look at:

- **Win rate with Wilson interval**, not just wins.
- **Crown differential** and **tower HP differential**.
- **Hog share** and **plays per match**: a policy that stops playing its win
  condition is not a better policy even if its score is high.
- **Illegal rate** must be `0.0%`.
- A gain only counts if it does not cost another opponent
  (`--eval-opponent brain,meta` selects on the worst).

### 4.1 Freeze a checkpoint for play

```powershell
New-Item -ItemType Directory -Force checkpoints\ppo_v2
Copy-Item tmp\rl\ppo_v2_best.pt checkpoints\ppo_v2\pilot_best_<step>.pt
Copy-Item tmp\rl\ppo_v2_best.json checkpoints\ppo_v2\pilot_best_<step>.json
Get-FileHash checkpoints\ppo_v2\pilot_best_<step>.pt -Algorithm SHA256
```

Record the step, git commit, hyperparameters, evaluation seed, opponent
results, and SHA-256 in a `manifest.json` next to the weights. Because
`checkpoints\` and `*.pt` are gitignored, keep a second copy outside the repo.

---

## 5. Live play

### 5.1 Friendly 1v1 (bot never presses Battle)

```powershell
.\run.ps1 -Brain rl -Checkpoint "tmp\rl\ppo_best.pt" -NoQueue
```

Start the friendly from your side. The bot decides it is in a game from the
screen, so it will play whatever it finds itself in. Without `-NoQueue` it
would keep tapping Battle in the lobby and pull itself into ladder between
friendlies.

### 5.2 Ladder

```powershell
.\run.ps1 -Brain rl -Checkpoint "tmp\rl\ppo_best.pt" -Matches 5
```

The bot presses Battle, plays up to five matches, then exits. `-Hours` caps the
wall-clock time (default `2.0`).

### 5.3 Useful variants

```powershell
# Rules brain instead of the learned policy
.\run.ps1 -Brain rules -Matches 5

# Rules brain without the local Ollama advisor
.\run.ps1 -Brain rules -NoAdvisor -Matches 5

# Keep the emulator at native resolution (sharper mirror/recording, slower capture)
.\run.ps1 -Brain rl -Checkpoint "tmp\rl\ppo_best.pt" -FullRes -NoQueue

# Long unattended loop: blocks of 5 matches, reviewed/tuned between blocks
.\run.ps1 -Brain rl -Checkpoint "tmp\rl\ppo_best.pt" -Forever

# Stop the bot and the supervisor
.\run.ps1 -Stop
```

`-Vision yolo` is the default, but this checkout has no YOLO detector weights
under `tmp\yolo\runs\`, so the bot logs `VISION unavailable ... staying on
buildabot` and uses the upstream ONNX detector. That is a working fallback, not
an error. To use YOLO, place weights at
`tmp\yolo\runs\<run>\weights\best.pt` and the side classifier at
`tmp\yolo\side_classifier.pt`. To suppress the warning, pass
`-Vision buildabot` explicitly.

### 5.4 Flags

| flag | meaning |
|---|---|
| `-Brain rules\|rl` | Hand-written policy, or simulator-trained checkpoint. |
| `-Checkpoint <path>` | Required for `-Brain rl`. Relative paths resolve from the repo root. |
| `-Matches N` | Match limit. Alias for the underlying `MatchCount`. |
| `-Hours H` | Wall-clock limit (default `2.0`). |
| `-NoQueue` | Never press Battle. Use for friendly 1v1. |
| `-FullRes` | Do not drop the emulator to 540×960. |
| `-Vision yolo\|buildabot` | Unit perception. |
| `-NoAdvisor` | Skip Ollama. Only affects `-Brain rules`. |
| `-Forever` | Supervisor loop instead of a fixed match count. |
| `-Watchdogs` | Re-enable persistent scheduled tasks with `-Forever`. Off by default. |
| `-Stop` | Kill `cr_bot.py` / `supervisor.py` and disable watchdogs. |

### 5.5 Logs and artefacts

```powershell
# Live decision log
Get-Content tmp\live\cr_bot.log -Wait -Tail 30

# Launcher output (display change, checkpoint loading, early errors)
Get-Content tmp\live\run_stdout.log -Wait -Tail 30

# Finished-match records
Get-ChildItem tmp\live\matches
```

Typical healthy startup:

```text
display set to 540x960 for capture speed; restored on exit.
playing 5 matches - brain: simulator-trained (ppo_best.pt)
VISION unavailable (no detector checkpoint found); staying on buildabot
BRAIN learned checkpoint ppo_best.pt trained to 301,056 steps, temperature=0.0
READY units=units.json(104) dry_run=False queue=False advisor=off
SCREEN None->lobby
```

The launcher sets the emulator to 540×960 for capture speed and restores it on
exit. If the process is killed hard, restore it manually:

```powershell
& $adb -s 127.0.0.1:16384 shell wm size reset
```

---

## 6. Troubleshooting

| symptom | cause / fix |
|---|---|
| `no interpreter at ...\.venvs\buildabot\Scripts\python.exe` | The venv is missing. Run the setup in section 1.2 or `scripts\bootstrap.ps1`. |
| `ModuleNotFoundError: clashroyalebuildabot` | `vendor\ClashRoyaleBuildABot` is missing or empty. Clone it at commit `f89ecbf` and reinstall into the venv. |
| `ModuleNotFoundError: torch` / `ultralytics` / `cv2` | The venv has BuildABot but not HastyCR's live extras. Run `uv pip install ... -e ".[live,dev]"`. |
| NumPy ABI / `_ARRAY_API not found` / binary incompatibility | Mixing system Python with the venv, or two NumPy versions. Use `.venvs\buildabot\Scripts\python.exe` for everything and reinstall both packages in that venv. |
| `emulator not found at 127.0.0.1:16384` | MuMu is not running, or the port changed. Run `MuMuManager.exe info -v all` and `adb connect`. |
| `refusing to run: ... is the CLASH OF CLANS instance` | Wrong serial. Do **not** bypass the guard; find the Clash Royale instance's port. |
| `Clash Royale is not in the foreground` | Open Clash Royale in the emulator before launching. |
| `VISION unavailable (no detector checkpoint found)` | Expected without YOLO weights. The bot uses BuildABot instead; this is not fatal. |
| `size mismatch for ...` when loading a checkpoint | The checkpoint's architecture/observation version does not match the current code. Use the code revision that trained it, or retrain. |
| `torch.cuda.is_available()` is `False` | CPU torch is installed. Live RL inference is fine; for GPU training install a CUDA build. |
| Bot sits at `SCREEN ...->lobby` | You passed `-NoQueue` (waiting for you to start a friendly), or the screen is a new layout the template does not recognise. Start the battle; if it stays `unknown`, inspect `tmp\live\cr_bot.log` and a screenshot. |
| Two bots/taps at once | Run `.\run.ps1 -Stop`, confirm no `cr_bot.py` process remains, then launch one bot only. |

---

## 7. What not to do

- Do not point the live bot at the Clash of Clans instance. The package check in
  `scripts\emulator.py` is there for a reason.
- Do not train on `--harvest-sprites` crops. They are labelled by our own
  detector; training on them teaches it its own mistakes.
- Do not reuse a training `--name` unless you intend to overwrite its
  `_best.pt` / `_last.pt` files.
- Do not treat a training eval or a 12-episode smoke test as a held-out result.
- Do not run two bots on the same emulator, and do not use `-NoQueue` for ladder.
- Do not delete `checkpoints\` or `tmp\rl\` expecting them to be recoverable
  from git; both are ignored.

---

## 8. Current verified state (2026-10-17)

- Venv: `.venvs\buildabot`, Python 3.12.10.
- BuildABot: pinned `f89ecbf309e698cb9fccdfb64a6ff2a930e6541e`.
- NumPy `2.4.6`, torch `2.14.1+cpu`, Ultralytics `8.4.178`,
  ONNX Runtime `1.26.0`.
- Emulator: MuMu instance 0, port `16384`, package
  `com.supercell.clashroyale` confirmed.
- Live checkpoint used: `tmp\rl\ppo_best.pt`, step `301,056`. Its sidecar
  `ppo_best.json` records `7W / 5L` on the eval that selected it; treat that as
  a smoke test and rerun section 4 before ladder.
- `tests\test_rl_policy.py`, `tests\test_emulator_capture.py`, and
  `tests\test_brain.py`: **108 passed, 2 skipped**.

---

## 9. File map

| path | role |
|---|---|
| `run.ps1` | Launcher: device check, resolution, checkpoint validation, `cr_bot.py`. |
| `scripts\cr_bot.py` | Live loop: capture, detect, decide, tap. |
| `scripts\emulator.py` | Device identity guard and port defaults. |
| `scripts\brain\rl_policy.py` | Checkpoint loading and live observation/action bridge. |
| `sim\env.py`, `sim\vecenv.py` | Simulator environment and parallel environments. |
| `sim\clone.py` | Behaviour cloning from the rule engine. |
| `sim\train_ppo.py` | PPO trainer and checkpoint writer. |
| `sim\train_tokens.py` | Token/transformer imitation trainer. |
| `scripts\rl_supervisor.py` | Unattended training, rollback and promotion. |
| `scripts\evaluate_pilot.py` | Fixed-seed held-out evaluation. |
| `scripts\head_to_head.py` | Two-checkpoint comparison. |
| `scripts\exploit_probe.py` | Degenerate-strategy probe. |
| `tmp\rl\` | Scratch training checkpoints. |
| `checkpoints\<name>\` | Promoted checkpoints and manifests. |
| `tmp\live\cr_bot.log` | Live decision log. |
| `tmp\live\rl_train.log` | Training log. |
| `reports\` | Evaluation and exploit reports. |
