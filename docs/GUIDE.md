# TutorState: how the codebase works

A guide to what each part of the repo does, how one case flows through the
agents, where data comes from and where outputs go, and how to run things.

---

## 1. The big picture

Three stages:

1. **Data**: conversations with expert labels, in `data/`.
2. **Runner**: `experiments/` feeds each conversation through the agents and
   saves what they produce.
3. **Evaluation**: `eval/` scores those saved outputs against the expert labels.

```
                         ┌──────────────── llm.py ────────────────┐
                         │ get_llm() → OpenRouter → Claude Haiku   │
                         │ cost tracking · $3 cap · key from .env  │
                         └───────────────────▲─────────────────────┘
                                             │ every LLM call
 ┌─────────── data/ ───────────┐   ┌──────── agents/ (B, C) ────────┐
 │ Bridge/…/441_test.json      │   │ analyser  memory  planner      │
 │ MrBench/test.json           │   │ generator  prompting (shared)  │
 └──────────────┬──────────────┘   └───────────────▲────────────────┘
                │ load_cases()                     │ one function per agent
                ▼                                  │
 ┌──────────────────── experiments/ (E) ───────────┴────────────────┐
 │ run_experiment (CLI) → runner (cases × conditions × seeds)       │
 │   → teacher_forced | free_running → turn (which agents are on)   │
 └──────────────┬───────────────────────────────────────────────────┘
                │ writes (gitignored)
                ▼
 results/<run>.jsonl  ·  results/trajectories/<run>.jsonl
                │
                ▼
 ┌──────────────────────── eval/ (D) ───────────────────────────────┐
 │ load: results + expert labels (joined only here)                 │
 │ diagnosis · strategy · leakage · judge → kappa · report          │
 └──────────────┬───────────────────────────────────────────────────┘
                ▼
 printed tables · results/<report>/report.json · confusion_*.png
```

**The agents never see the expert labels.** The data loader leaves them out,
and they are only joined back in by `eval/load.py`, after the agents have run.

---

## 2. Folder map

| Path | Owner | What it is |
|---|---|---|
| `data/Bridge/bridge_data_size_441/`, `..._142/` | A | BRIDGE conversations + expert labels (train/validation/test JSON). The two versions were split separately: never mix them. |
| `data/MrBench/` | A | MRBench conversations, reference solutions and human ratings, plus `dataset_utils.py` to read them |
| `schemas.py` | all | The shared field definitions: the 7 actions, the 7 error types, and the shapes of the analysis, memory and planner output |
| `llm.py` | all | One gateway to the model. `get_llm()` returns the group model via OpenRouter, with cost tracking and a $3 cap per run |
| `prompts/*.txt` | B, C, D | Every prompt, as plain text, so prompt edits never touch code |
| `agents/` | B, C | The four agents, one function each |
| `experiments/` | E | The runner: loads cases, runs conditions × seeds, saves results |
| `eval/` | D | Scoring, leakage check, LLM judge, kappa, hand-scoring sheets, report |
| `mocks/`, `scripts/`, `tests/` | each | Fake inputs, try-out scripts, offline tests |
| `requirements.txt` | all | `pip install -r requirements.txt` |
| `.env` | you | `OPENROUTER_API_KEY=...`. Gitignored; **never commit the key**. |

---

## 3. What happens to one case

Example: a history of `Tutor, Student, Tutor, Student`, run in the **full** condition.

```
 history: [T: "What is the area?"] [S: "4"] [T: "Not quite"] [S: "4 m"]
                                     │                         │
          earlier student turn ──────┘           final turn ───┘
                │                                     │
   build_prior_memory()                         run_turn()
   (memory + full only)                               │
                │                                     ▼
                ▼                              ┌─ Analyser ─┐  sees whole history
   ┌─ Analyser ─┐ sees history up to "4"       └─────┬──────┘
   └─────┬──────┘                                    ▼
         ▼                                     ┌─ Memory ───┐  prior memory + new analysis
   ┌─ Memory ───┐ empty → attempts=1  ────────▶└─────┬──────┘  → attempts=2, occurrences=2
   └────────────┘                                    ▼
                                               ┌─ Planner ──┐  → {reason, action: "EXPLAIN"}
                                               └─────┬──────┘
                                                     ▼
                                               ┌─ Generator ┐  → "Area means length × width…"
                                               └─────┬──────┘
                                                     ▼
                       one results row: analysis · memory · planner · response · error
```

Each earlier student turn is analysed using **only the conversation up to that
turn**, so no later information leaks in.

### The four ablation conditions

These are the same picture with boxes switched off (`experiments/conditions.py`):

| Condition | Earlier turns → memory | Analyser | Memory | Planner | Generator | LLM calls* |
|---|---|---|---|---|---|---|
| baseline | – | – | – | – | ✓ | 1 |
| analyser | – | ✓ | – | – | ✓ | 2 |
| memory | ✓ | ✓ | ✓ | – | ✓ | 2 + earlier turns |
| full | ✓ | ✓ | ✓ | ✓ | ✓ | 3 + earlier turns |

\*Memory never calls the LLM.

The generator uses **one prompt template** in every condition. Inputs a
condition doesn't have are simply left out, so the conditions differ only in
what information they get, never in prompt wording.

---

## 4. The agents

All four follow the same pattern:
- The prompt comes from `prompts/*.txt`, and the input text is built by `agents/prompting.py`.
- The LLM is told the exact JSON shape to return, and the output is checked against it.
- If the output is invalid, the agent retries once, then raises an error. **There is no silent default.**

### Analyser: "what did the student just do?"
`agents/analyser.py` (Track B), prompts `analyser_system.txt` and `error_type_definitions.txt`

| | |
|---|---|
| In | history, lesson topic (and problem, if any) |
| LLM returns | `reasoning` first (it labels more accurately after reasoning), then the labels |
| Checks | `error_type` is one of the 7; `confidence` is low/medium/high; `no_error` forces `correct=true` |
| Out | `{correct, concept, error_type, misconception, confidence}` |
| Error | `AnalyserError` |

The `reasoning` is **dropped** before output, because it often states the
answer and the generator sees the analysis. "ok", "brb" and similar → `correct=null`, `error_type="diagnose"`.

### Memory: "what do we know about this student overall?"
`agents/memory.py` (Track B). **No LLM, just rules:**

- An actual attempt adds 1 to `attempts`.
- A correct answer adds 1 to `correct_attempts` and marks that concept's misconceptions "resolved".
- A wrong answer finds the matching misconception by word overlap, or adds it, and counts it:
  - seen once = "tentative";
  - seen twice or more = "recurring".
- Non-attempts (`correct=null`) change no counts.
- `mastery` comes from the correct-answer rate, and is low while a recurring misconception is active.
- `hints_given` is counted from tutor messages by keyword.

**Why rules and not an LLM?** Counting is bookkeeping: code does it perfectly,
for free, the same way every seed, and it can be tested against scripted
students. The judgement (which misconception) is already made by the
analyser's LLM call.

**The trade-off:** word-overlap matching treats paraphrases ("adds the bottoms"
vs "adds denominators") as different, and hint counting is a keyword guess.

### Planner: "what teaching move next?"
`agents/planner.py` (Track C), prompts `planner_system.txt` and `action_definitions.txt`

| | |
|---|---|
| In | history, topic, and the analysis and memory when present |
| Rules in prompt | low confidence leans PROBE; repeated hints that haven't helped → EXPLAIN or SIMPLIFY; correct → AFFIRM |
| LLM returns | `{reason, action}`; `reason` comes first, and the action is limited to the 7 by the schema |
| Out | `{"reason": "...", "action": "EXPLAIN"}` |
| Error | `PlannerError` (never a default action, which would bias the confusion matrix) |

### Generator: "say it to the student"
`agents/generator.py` (Track C), prompt `generator_system.txt`

| | |
|---|---|
| In | history and topic, plus whatever is on: analysis, memory, action + reason, reference solution (MRBench) |
| Rules | 1–3 short, warm sentences for a primary-school student; no invented problem details; never reveal the answer unless the action is CORRECT; with an action, make exactly that move |
| Out | a plain-text message, with "Tutor:" labels and quotes stripped |
| Error | `GeneratorError` |

### Prompt builder
`agents/prompting.py` is shared by all three LLM agents:
- loads the prompt files;
- formats the history as `Tutor: … / Student: …`;
- formats analysis and memory as bullets, with lists as nested bullets;
- assembles the sections in order: Context → Conversation → Diagnosis → Student state → Teaching move.

---

## 5. The runner (`experiments/`)

| File | What it does |
|---|---|
| `run_experiment.py` | Command-line entry point: options → `load_cases` → `get_llm` → `run_experiment`, then prints spend |
| `dataloader.py` | `load_cases()` detects BRIDGE or MRBench and returns cases `{case_id, history, lesson_topic, problem, reference_solution}` with no expert labels. For MRBench it splits the conversation string into turns and uses `conversation_id` as the case ID. |
| `runner.py` | Loops cases × conditions × seeds, skips rows already done, catches failures, writes one row per case, prints the failure rate |
| `turn.py` | `build_prior_memory()` handles earlier student turns; `run_turn()` runs the final turn with the condition's switches |
| `teacher_forced.py` | Replay mode: the dataset's real history, then one tutor reply. **Main results use this.** |
| `free_running.py` | Loops with a simulated student, to test multi-turn behaviour. **The simulator isn't built yet.** |
| `adapters.py` | Thin wrappers linking the runner to the agents |
| `conditions.py` | The four on/off combinations |
| `io.py` | Appends JSON lines; works out which keys already succeeded |

### When a case fails

- **Agent, LLM and API errors** (invalid output, refusal, rate limit, timeout):
  - the case is saved with an `error` field and the run continues;
  - **rerunning the same command retries only the failed rows**;
  - `eval.report` shows the failure rate per condition, and failed rows are left out of every score.
- **These still stop the run:**
  - `BudgetExceeded`, when the spend cap is hit;
  - bugs in our own code, such as a `KeyError` or `TypeError`.
- Rerun until failures are close to 0. If a condition stays above about 2%, look at its error types before reporting.

---

## 6. Evaluation (`eval/`)

| File | What it does |
|---|---|
| `load.py` | Reads results (latest success wins; failures kept for the rate) and expert labels; builds what the judge sees |
| `metrics.py` | Scoring: correct if it matches **any** expert, macro-F1, confusion matrices, bootstrap CIs, paired comparisons |
| `diagnosis.py` | Analyser `error_type` vs BRIDGE `e_clean` |
| `strategy.py` | Planner `action` vs BRIDGE `z_what_clean`; optional lenient sets of acceptable actions |
| `leakage.py` | Does a response contain the final answer? MRBench only, since BRIDGE has no solutions |
| `judge.py` | LLM ratings on the 8 MRBench dimensions, cached by prompt version (`python -m eval.judge`) |
| `kappa.py` | Judge-vs-human agreement |
| `human_sheet.py` | Blind CSV sheets for hand-scoring, and their agreement |
| `report.py` | One command that prints every table (failures, diagnosis, strategy, comparisons, leakage, judge) and saves `report.json` and charts |

---

## 7. Where things are stored

| What | Where | In git? |
|---|---|---|
| Input data | `data/Bridge/...`, `data/MrBench/...` | yes |
| Prompts | `prompts/*.txt` | yes |
| API key | `.env` (`OPENROUTER_API_KEY=...`) | **no**, gitignored |
| Run results (one line per case × condition × seed) | `results/<dataset>_<split>_<mode>_<provider>_<model>.jsonl`, or your `--out` | no, `results/` is gitignored |
| Step-by-step trajectories (incl. earlier-turn analysis) | `results/trajectories/<same name>.jsonl`, or your `--trajectory-out` | no |
| Judge ratings | the `--out` you give `eval.judge` | no |
| Report | `--out-dir`: `report.json`, `confusion_*.png` | no |
| Hand-scoring sheets | the `--out` you give `eval.human_sheet` | no |

Each results row contains: `dataset, split, case_id, condition, seed, mode,
model, analysis, memory, planner, response, error`.

---

## 8. Model and budget

| Setting | Value |
|---|---|
| Group model | `anthropic/claude-haiku-5.5` via OpenRouter ($0.10 in / $0.50 out per million tokens) |
| Settings | at the top of `llm.py`: `MODEL`, `EFFORT="low"`, `MAX_TOKENS=4000`, `MAX_SPEND_USD=3.0` |
| How to use it | always `from llm import get_llm; llm = get_llm()` |

**Rules:**
- Agree with the group before changing `MODEL`, because it changes everyone's results.
- For a one-off model, pass `--model <openrouter slug>`.

**Measured costs:**

| What | Cost |
|---|---|
| One LLM call | about $0.00015 |
| One case through all 4 conditions | about $0.0017 |
| Full 441-test experiment (66 cases × 4 conditions × 3 seeds) | about $0.35 |
| Judging those runs | about $0.50 |
| A 3-case trial | about $0.005 |

**Free models** (`...:free`) cost $0, but are limited to 50 requests a day
(1,000 a day once an account has bought $10+ of credit). They're fine for
checking that the pipeline works. **Never report results from them or tune
prompts on them**, because a different model makes different choices.

**Claude Pro** doesn't cover API calls. `ClaudeLLM` (`--provider anthropic`)
needs separate pay-as-you-go credits from console.anthropic.com.

---

## 9. How to run things

### First-time setup
```powershell
pip install -r requirements.txt
# create .env in the repo folder containing one line:
#   OPENROUTER_API_KEY=sk-or-v1-...
python -m pytest                                    # offline tests, free
```

To use your **own** OpenRouter key for one terminal session, without touching
`.env` (the environment variable wins over `.env`):
```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-<personal key>"
```

### Trial run: 3 cases × 4 conditions × 1 seed, about $0.005
```powershell
python -m experiments.run_experiment `
    --data data/Bridge/bridge_data_size_441/441_validation.json `
    --dataset-name trial --split validation --seeds 0 --max-cases 3 `
    --out results/trial.jsonl --trajectory-out results/trial_traj.jsonl
```

Read the outputs case by case (conversation, each condition's diagnosis, memory, action and reply, and the experts):
```powershell
python scripts/show_results.py results/trial.jsonl --data data/Bridge/bridge_data_size_441/441_validation.json
```

See the score tables:
```powershell
python -m eval.report --results results/trial.jsonl `
    --bridge data/Bridge/bridge_data_size_441/441_validation.json `
    --order baseline,analyser,memory,full
```

**Notes:**
- Rerunning the same command skips cases that already succeeded and retries failures.
- To start fresh, delete the `--out` file.
- `--conditions full` runs one condition only.
- `--seeds 0 1 2` runs three replicates.

### Larger development runs (never on test)

Development runs and prompt tuning use only histories outside every candidate
reporting set (142 validation, 142 test, 441 test). Build those files once:
```powershell
python scripts/make_dev_set.py     # -> results/dev/dev_train.json (281), dev_validation.json (59)
```
Then run the whole development validation set, all 4 conditions, 1 seed (about $0.10):
```powershell
python -m experiments.run_experiment --data results/dev/dev_validation.json `
    --dataset-name dev441 --split validation --seeds 0 `
    --out results/dev_val.jsonl --trajectory-out results/dev_val_traj.jsonl
python -m eval.report --results results/dev_val.jsonl --bridge results/dev/dev_validation.json `
    --order baseline,analyser,memory,full --out-dir results/dev_val_report
```
**Test sets are run once, at the end,** after prompts are frozen and the group
has agreed which sets to report. Every look at test results makes them less trustworthy.

### MRBench trial
```powershell
python -m experiments.run_experiment --data data/MrBench/test.json `
    --dataset-name mrbench --split test --conditions full --seeds 0 --max-cases 2 `
    --out results/trial_mr.jsonl --trajectory-out results/trial_mr_traj.jsonl
python scripts/show_results.py results/trial_mr.jsonl --data data/MrBench/test.json
```
Add `--include-solution` to give the tutor the reference solution; it's off by default, and this is a group decision.

### Judge
```powershell
# check the judge against MRBench's human ratings first (~$0.06 for 100)
python -m eval.judge mrbench --split train --n 100 --out results/judge_mrbench_train.jsonl
python -m eval.kappa results/judge_mrbench_train.jsonl

# then judge a run, and include it in the report
python -m eval.judge results --results results/trial.jsonl `
    --bridge data/Bridge/bridge_data_size_441/441_validation.json --out results/judge_trial.jsonl
python -m eval.report --results results/trial.jsonl `
    --bridge data/Bridge/bridge_data_size_441/441_validation.json --judge results/judge_trial.jsonl
```
Add `--model <slug>` to use a different judge model, for example one from a different family than the agents.

### Try out a single track's agents
```powershell
python scripts/try_planner_generator.py      # Track C on 5 mock cases (~$0.002)
python scripts/try_analyser.py --n 20        # Track B's analyser on train cases
```

---

## 10. Status and open issues

**Working:**
- all 4 conditions run end to end on BRIDGE and MRBench;
- memory is built from earlier student turns;
- failures are recorded and retried;
- the judge has a command-line entry point;
- costs are tracked.

**Still open:**
- **Simulated student** for free-running mode. Lower priority; the scripted
  memory-consistency profiles (fixed student messages, known correct memory) are more useful.
- **"teacher_forced" naming:** the runner's mode means *replay the dataset*, but
  the proposal's teacher-forcing means *feed in gold labels*. Suggested: rename it
  `replay`, and add gold-diagnosis and gold-action variants.
- **Reporting sets:** the 142 and 441 versions overlap across splits. Agree one
  pool of histories for tuning that excludes every reporting set.
- **Track D:**
  - action labels for the non-planner conditions;
  - leakage broken down by action (CORRECT reveals the answer by design);
  - choice of judge model (risk of a model favouring itself).
- **Track C:**
  - the planner over-uses PROBE;
  - the generator sometimes says it "can't see the problem" (BRIDGE histories
    don't include it). Fix this in `generator_system.txt`.
- `--split` isn't checked against `--data`.
