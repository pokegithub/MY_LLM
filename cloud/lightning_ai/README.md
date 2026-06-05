# Lightning AI Free-Credit Cloud Prep

This folder prepares safe Lightning AI cloud smoke execution for MY_LLM. It does not start Phase B, train a model, run a full 7B/14B bakeoff, deploy serving, or prove model quality.

## Safety Rules

- Use only the user's free Lightning AI credits.
- Hard budget: 15 credits maximum.
- Paid overage is blocked.
- Dry-run is the default.
- Execute mode requires explicit confirmation environment variables.
- Do not give Codex your Lightning AI password.
- Do not store passwords, tokens, cookies, payment data, or credentials in this repo.
- If Lightning authentication is missing or credit balance cannot be verified, execute mode fails closed.

## Manual Authentication

Log in manually through the Lightning AI UI or the official Lightning CLI. Codex should not receive your password.

If you use a token, provide it only through environment variables or Lightning's own secret system. Do not commit it, paste it into scripts, or write it into config files.

For the local runner to treat authentication as manually confirmed, set:

```powershell
$env:MYLLM_LIGHTNING_AUTH_CONFIRMED = "YES"
```

This is only a confirmation flag. It is not a credential.

## Free-Credit Verification

Before execute mode, manually verify your current free-credit balance in the Lightning AI UI. The runner does not invent prices or balances.

Required execute-mode budget variables:

```powershell
$env:MYLLM_CLOUD_CONFIRM_EXECUTE = "YES"
$env:MYLLM_CLOUD_MAX_CREDITS = "15"
$env:MYLLM_LIGHTNING_FREE_CREDITS_CONFIRMED = "YES"
$env:MYLLM_LIGHTNING_PRICE_CONFIRMED = "YES"
```

If you know the numeric values, prefer:

```powershell
$env:MYLLM_LIGHTNING_FREE_CREDITS_AVAILABLE = "15"
$env:MYLLM_LIGHTNING_ESTIMATED_CREDITS = "3"
```

The estimate must stay within the free-credit balance and the 15-credit hard maximum.

## GPU Policy

Use the cheapest compatible GPU, not blindly the cheapest GPU. The absolute cheapest GPU can fail if it lacks enough VRAM/CUDA capability for the smoke task, which wastes time and possibly credits. Prices must come from the Lightning UI or an authenticated Lightning source; unknown prices block execute mode.

Policy file:

```text
cloud/lightning_ai/gpu_policy_v1.json
```

## Dry Run

PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File cloud\lightning_ai\run_lightning_candidate_readiness.ps1 -DryRun
```

Direct Python:

```powershell
.\.venv\Scripts\python.exe cloud\lightning_ai\cloud_readiness_runner.py --dry-run
.\.venv\Scripts\python.exe cloud\lightning_ai\cloud_readiness_runner.py --check-env
.\.venv\Scripts\python.exe cloud\lightning_ai\cloud_readiness_runner.py --print-plan
```

Dry-run may report `manual_login_required` or unknown credits. That is expected until you manually log in and confirm the free-credit balance.

## Execute Mode

Execute mode is blocked unless all guards pass:

- `--execute` is provided.
- `MYLLM_CLOUD_CONFIRM_EXECUTE=YES`.
- `MYLLM_CLOUD_MAX_CREDITS=15`.
- Auth is manually confirmed.
- Free-credit balance is known or manually confirmed.
- Estimated cost/price is known or manually confirmed.
- Selected task class is allowed.
- Task fits the declared policy.

PowerShell execute shape:

```powershell
powershell -ExecutionPolicy Bypass -File cloud\lightning_ai\run_lightning_candidate_readiness.ps1 -Execute -TaskClass candidate_readiness_smoke
```

Do not run execute mode until you have verified free credits and GPU pricing manually.

## Allowed Commands

- `python run.py status`
- `python run.py deps`
- `python run.py compile-source`
- `python run.py repo-assist-eval`
- `python run.py hidden-eval-seed-run`
- `python run.py trajectory-quality-audit`
- `python run.py candidate-readiness-smoke`
- `python run.py agent-backend-smoke`

## Blocked Commands

- `train`
- `sft`
- `dpo`
- `distill`
- `improve`
- `rlvr`
- `full_bakeoff`
- production serving loops

## Stop a Job

Use the Lightning AI UI to stop a running job immediately. If using an official Lightning CLI, use the provider's documented stop/cancel command for the specific job id. Do not leave smoke jobs running unattended.

## Reports

Runner reports are written under:

```text
run_artifacts/cloud/lightning_ai/
```

Reports must not contain passwords, tokens, cookies, payment data, or secrets.
