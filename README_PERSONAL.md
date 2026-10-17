# My Hasty-CR Training Notes

Full reference: [`docs/TRAINING_AND_RUNBOOK.md`](docs/TRAINING_AND_RUNBOOK.md)

## 1. Open the venv

```powershell
cd C:\Users\jaid2\OneDrive\Documents\GitHub\Hasty-CR
.\.venvs\buildabot\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venvs\buildabot\Scripts\Activate.ps1
```

Your prompt should show `(buildabot)`. Check it:

```powershell
python --version
```

Without activation, use `.\.venvs\buildabot\Scripts\python.exe` instead of `python`.

## 2. Before training

Stop the live bot so it does not compete for CPU:

```powershell
.\run.ps1 -Stop
```

Check free disk space (keep at least 10 GB):

```powershell
Get-PSDrive C | Select-Object Used,Free
```

Use a **new `--name`** for each run, and make sure `--init` points to a real checkpoint.

## 3. Run the supervisor

Start a fresh run from the behaviour clone:

```powershell
python scripts\rl_supervisor.py `
  --name ppo_night1 `
  --hours 6 `
  --envs 8 `
  --init tmp\rl\clone.pt `
  --opponent brain `
  --brain-share 0.5 `
  --eval-opponent brain `
  --anchor tmp\rl\clone.pt `
  --anchor-coef 0.1
```

To continue your current policy instead:

```powershell
python scripts\rl_supervisor.py `
  --name ppo_night2 --hours 6 --envs 8 `
  --init tmp\rl\ppo_best.pt `
  --opponent brain --brain-share 0.5 --eval-opponent brain
```

On CPU torch, use `--envs 4`.

## 4. Monitor and stop

```powershell
Get-Content checkpoints\ppo_night1\supervisor.log -Wait -Tail 30
```

Press **Ctrl+C** in the training window to stop it.

Force-stop training if needed:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*rl_supervisor.py*' -or $_.CommandLine -like '*train_ppo*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

## 5. Results

| path | contents |
|---|---|
| `checkpoints\ppo_night1\best.pt` | Best trained checkpoint. |
| `checkpoints\ppo_night1\SUPERVISION.md` | Final report and verdict. |
| `checkpoints\ppo_night1\supervisor.log` | Training decisions and evals. |

Read the report first:

```powershell
notepad checkpoints\ppo_night1\SUPERVISION.md
```

## 6. Test and play

```powershell
python scripts\evaluate_pilot.py `
  --ckpt checkpoints\ppo_night1\best.pt --episodes 200 --seed 9000

.\run.ps1 -Brain rl -Checkpoint "checkpoints\ppo_night1\best.pt" -NoQueue
```

## 7. Quick fixes

- **Activation blocked** → run the `Set-ExecutionPolicy -Scope Process` command above.
- **`no init checkpoint`** → pass a real path, e.g. `--init tmp\rl\clone.pt`.
- **`ModuleNotFoundError`** → the venv is not active; activate it or use the full `.venvs\buildabot\Scripts\python.exe` path.
- **Training is slow** → CPU torch is installed; lower `--envs` or install a CUDA torch build.
- **Do not reuse `--name`** unless you intend to write into the same `checkpoints\<name>\` folder.
