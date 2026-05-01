# CPS Case Studies

This repository contains three cyber-physical system case studies:

- Robot navigation
- Industrial valve control
- Building access control

Each case study evaluates multiple decision systems against clean and adversarial text instructions, then writes detailed CSV results and summary files.

## Setup

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the Python packages used by the runners and analysis scripts:

```powershell
pip install torch transformers pandas pytest
```

The LLM-backed systems use Hugging Face model IDs from the runner scripts. The first run may download models and can take a while. If a model requires authentication, log in with Hugging Face before running:

```powershell
huggingface-cli login
```

## Run All Three Case Studies

From the repository root, run:

```powershell
python run_robot_all_systems.py --dataset robot_text_attack_families_1000.json
python run_valve_all_systems.py --dataset valve_text_attack_families_1000.json
python run_building_all_systems.py --dataset building_text_attack_families_1000.json
```

Each command runs all configured systems for that case study:

- `abci`
- `deterministic_grammar_policy`
- `llm_assist_rule_approval`
- `direct_decision`
- `role_separated`
- `schema_constrained`

## Run One System Only

Use `--system` to run a single system:

```powershell
python run_robot_all_systems.py --dataset robot_text_attack_families_1000.json --system deterministic_grammar_policy
python run_valve_all_systems.py --dataset valve_text_attack_families_1000.json --system abci
python run_building_all_systems.py --dataset building_text_attack_families_1000.json --system schema_constrained
```

Add `--show-all` if you want every case printed to the console instead of only failures or flips:

```powershell
python run_robot_all_systems.py --dataset robot_text_attack_families_1000.json --show-all
```

## Outputs

The runner scripts write results in the repository root.

Detailed per-system CSV files follow this pattern:

```text
robot_<system>_<model>_<time>.csv
valve_<system>_<model>_<time>.csv
building_<system>_<model>_<time>.csv
```

Summary files follow this pattern:

```text
robot_all_systems_summary_<time>.csv
robot_all_systems_summary_<date>_<time>.txt
valve_all_systems_summary_<time>.csv
valve_all_systems_summary_<date>_<time>.txt
building_all_systems_summary_<time>.csv
building_all_systems_summary_<date>_<time>.txt
```

## Analyze Results

After a run, pass one or more detailed per-system CSV files to the matching analysis script:

```powershell
python analyze_robot_results.py --inputs robot_*.csv --output-prefix robot_analysis
python analyze_valve_results.py --inputs valve_*.csv --output-prefix valve_analysis
python analyze_building_results.py --inputs building_*.csv --output-prefix building_analysis
```

To compare all systems against a baseline label, use `--compare-to`:

```powershell
python analyze_robot_results.py --inputs robot_*.csv --compare-to "abci | Qwen/Qwen3-1.7B" --output-prefix robot_analysis
```

Analysis scripts create:

- `<output-prefix>_summary.csv`
- `<output-prefix>_comparisons.csv`

## Run Tests

Run the unit tests with:

```powershell
pytest
```
