# Adaptive Utility-Weighted SFT

This repo implements a small, reproducible first version of an adaptive data-selection experiment for supervised fine-tuning.

The claim being tested is simple: during SFT, the usefulness of a batch changes as the model changes. A lightweight evaluator trained to predict validation-gradient alignment should select better training batches than random SFT and fixed static filters.

This is SFT only. There is no RLHF, DPO, reward model, preference data, or example-level adaptive weighting in V1.

## Install

```bash
cd adaptive_sft
CONDA_PKGS_DIRS="$PWD/.conda-pkgs" conda create -y -p "$PWD/.conda-env" python=3.10 pip
PYTHONNOUSERSITE=1 PIP_CACHE_DIR="$PWD/.pip-cache" ./.conda-env/bin/python -m pip install -r requirements.txt
```

The runner automatically uses `./.conda-env/bin/python` when it exists, then `./venv/bin/python`, then system `python`.

`requirements.txt` pins PyTorch to the CUDA 12.8 wheel (`torch==2.11.0+cu128`), matching nodes whose `nvidia-smi` reports CUDA 12.8. If GPUs are visible but PyTorch cannot initialize CUDA, the training code now fails fast with a wheel/driver message instead of silently running the model on CPU. Set `ADAPTIVE_SFT_ALLOW_CPU=1` only when a deliberately slow CPU run is intended.

The default model is `Qwen/Qwen2.5-0.5B` and the default dataset is `AI-MO/NuminaMath-CoT` using the `train` split. The dataset contains math reasoning problem/solution pairs in chat-style `messages`, so the experiment remains assistant-only SFT on reasoning traces.

If Hugging Face access needs a token, export `HF_TOKEN` before running or place it in a gitignored local env file:

```bash
printf 'HF_TOKEN=<your-token>\n' > .env.local
chmod 600 .env.local
```

The bash launchers source `.env.local` / `.env` and also export `HUGGING_FACE_HUB_TOKEN` from `HF_TOKEN`.

## Smoke Test

Smoke mode is intentionally small:

- train pool: 1,000 examples
- utility validation: 100 examples
- test: 100 examples
- warmup: 20 optimizer steps
- continuation: 50 optimizer steps
- adaptive candidates: `K=4`

Run it with:

```bash
bash scripts/run_all.sh smoke
```

The smoke run also writes `outputs/smoke/masking_sanity.txt`, which decodes one processed example and prints the full text, active assistant label spans, active label token count, and ignored token count.

To rerun only the adaptive methods against existing smoke artifacts:

```bash
bash scripts/run_all_multi_gpu.sh adaptive-smoke
```

## Full Experiment

Full mode always runs smoke mode first and aborts if smoke fails:

```bash
bash scripts/run_all.sh full
```

Full mode writes to `outputs/`; smoke mode writes to `outputs/smoke/`. The mandatory full-run smoke preflight also writes to `outputs/smoke/` by default, even when `ADAPTIVE_SFT_OUTPUT_DIR` is set for the full run, so smoke data cannot contaminate the full data cache. Set `ADAPTIVE_SFT_SMOKE_OUTPUT_DIR` to override that preflight location.

Noise stress-test modes corrupt only the training pool and keep utility validation/test clean:

```bash
bash scripts/run_all.sh smoke_noise
bash scripts/run_all.sh full_noise
```

For a single node with multiple GPUs, use the multi-GPU launcher:

```bash
GPU_IDS=0,1,2,3 bash scripts/run_all_multi_gpu.sh full
```

This still runs data prep and evaluator training once. Warmup and utility-label collection expose all selected GPUs and enable PyTorch `DataParallel`; independent continuation methods then run across GPUs:

- GPU 0: `random_sft`, then `static_loss_sft`
- GPU 1: `static_gradient_sft`, then `oracle_gradient_sft`
- GPU 2: `adaptive_utility_sft`
- GPU 3: `adaptive_shuffled_scores`

Per-method launcher logs are written under `outputs/launcher_logs/` or `outputs/smoke/launcher_logs/`.

To rerun only `adaptive_utility_sft` and `adaptive_shuffled_scores` against existing full artifacts:

```bash
GPU_IDS=0,1 bash scripts/run_all_multi_gpu.sh adaptive
```

## What Runs

1. Prepare NuminaMath-CoT data with assistant-only label masking.
2. Train a shared random-SFT LoRA warmup checkpoint.
3. Collect batch-level utility labels using cosine similarity between candidate-batch LoRA gradients and an averaged utility-validation LoRA gradient.
4. Train a tiny MLP evaluator on numeric batch features only. By default, raw utility labels are normalized within each collection step so the evaluator learns candidate ordering for a model state.
5. Train:
   - `random_sft`
   - `static_loss_sft`
   - `static_gradient_sft`
   - `adaptive_utility_sft`
   - `adaptive_shuffled_scores`
   - `oracle_gradient_sft`
6. Run diagnostics:
   - evaluator imitation of true gradient utility in candidate-set selection
   - direct gradient-cosine vs one-step validation-loss improvement sanity check
7. Evaluate all methods with identical held-out SFT loss code.
8. Plot losses, selection dynamics, overhead, utility-label histogram, evaluator scatter, imitation diagnostics, and gradient-signal sanity.

## Methodology For New Readers

This experiment asks one question: can SFT improve if the training loop chooses the next batch more carefully than random sampling? Every method trains the same LoRA-adapted language model with the same SFT loss. The methods differ only in how they choose the next batch.

The full data split is:

- `train_pool`: examples that methods may train on
- `utility_val`: held-out examples used only for batch-selection signals and validation curves
- `test`: held-out examples used only for final test loss

The model is trained with ordinary next-token SFT on assistant responses. The user prompt is part of the input context, but user, system, and padding tokens are masked out with label `-100`, so only assistant tokens contribute to the loss.

### Step 1: Shared Warmup

The code first loads `Qwen/Qwen2.5-0.5B`, attaches LoRA adapters, and trains those adapters for `training.warmup_steps` random SFT steps. This creates one shared `warmup` checkpoint. All continuation methods start from this exact same checkpoint, which keeps the comparison focused on batch selection rather than different starting weights.

### Step 2: Utility Labels

During normal training, a gradient is the update direction produced by a batch. If a candidate train batch gives a gradient pointing in a similar direction to a validation batch gradient, the experiment treats that candidate as useful.

For each utility-label collection state, the code:

1. Samples several batches from `utility_val`.
2. Computes LoRA gradients for those validation batches.
3. Averages them into one validation gradient.
4. Samples candidate batches from `train_pool`.
5. Computes each candidate batch's LoRA gradient.
6. Stores the cosine similarity between candidate gradient and validation gradient as the candidate's utility label.

In formula form:

```text
utility_label(candidate_batch) =
  cosine_similarity(candidate_train_gradient, utility_validation_gradient)
```

A high positive utility label means the train batch would update LoRA weights in a direction similar to the validation set. A near-zero label means the candidate is mostly unrelated to the validation direction. A negative label means the candidate points against that validation direction. These gradients are computed only over trainable LoRA parameters, not all base-model weights.

By default, raw utilities are normalized within each collection step:

```text
step_zscore_utility = (utility - mean_utility_at_this_step) / std_utility_at_this_step
```

That target teaches the evaluator to rank candidate batches within the same model state, instead of comparing raw utility magnitudes across different training states.

### Step 3: Evaluator

The evaluator is a small MLP, separate from Qwen. It is trained on records shaped like:

```text
batch_features -> utility_label
```

The default features are cheap batch statistics:

- mean SFT loss
- per-example loss standard deviation
- mean input length
- max input length
- mean assistant-response length
- mean token entropy
- mean label confidence
- training progress fraction

The evaluator's job is to predict which candidate batch is likely to have high gradient utility without computing candidate gradients online. If its dev ranking accuracy, Spearman, or imitation metrics are close to random, then adaptive selection is weak even if the SFT training loop itself is working.

### Step 4: Continuation Methods

After warmup, every method restarts from the same warmup checkpoint and runs `training.continuation_steps` normal SFT optimizer steps. The difference is the batch-selection rule.

`random_sft` samples one random training batch per step and trains on it. This is the baseline for ordinary random SFT.

`static_loss_sft` scores a fixed pool of candidate batches once at the warmup checkpoint using current SFT loss. The default `middle-loss` strategy keeps batches near the middle of the loss distribution, builds a fixed schedule from that selected pool, and trains on that schedule for all continuation steps.

`static_gradient_sft` computes one utility-validation gradient at the warmup checkpoint, scores a fixed pool of train batches by gradient cosine utility, selects the highest-scoring fraction, and trains on that fixed selected pool. It uses gradient utility, but only once before continuation training begins.

`adaptive_utility_sft` performs online batch selection. At every continuation step it samples `K` candidate batches, computes cheap features for each candidate under the current model, uses the evaluator to score them, standardizes the scores, samples one candidate with softmax temperature, and then performs one normal SFT update on the selected batch.

`adaptive_shuffled_scores` is a control for `adaptive_utility_sft`. It computes the same candidate features and evaluator scores, but shuffles the scores before selecting. This keeps the same score distribution and non-uniform softmax behavior while breaking the link between a candidate and its evaluator score. If `adaptive_utility_sft` beats this method, the evaluator's ranking is adding useful information.

`oracle_gradient_sft` is the expensive upper bound. At every continuation step it samples `K` candidate batches, computes the real validation gradient and real candidate gradients, selects the candidate with the best true cosine utility, and trains on that batch. If this method cannot beat `random_sft`, the gradient-utility signal itself is probably not useful in the current setup.

### Step 5: Evaluation

After continuation training, the evaluation code loads each final method checkpoint and computes SFT loss on the held-out `test` split. The main comparison is final `test_loss` and `test_perplexity`, with validation curves and diagnostics used to understand why a method did or did not work.

The most important mental model is:

```text
same model + same SFT update + same number of continuation steps
different rule for choosing the next training batch
```

## Assistant-Only Loss

Processed labels use `-100` for user, system, and padding tokens. Only assistant response tokens are active labels. If a tokenizer chat template is available, the data processor uses it and masks only assistant content spans; otherwise it falls back to:

```text
User: ...
Assistant: ...
```

## Static Loss Baseline

`static_loss_sft` defaults to `middle-loss`, which selects a fixed set of middle-loss candidate batches from the warmup checkpoint. `top-loss` is available:

```bash
python -m src.train_baseline_static_loss --strategy top-loss
```

The static baselines expand their scored candidate pools when needed so the selected batch pool can cover the requested continuation steps. This keeps them static filters without repeatedly training on a tiny selected subset.

## Oracle And Diagnostics

`oracle_gradient_sft` is an expensive upper-bound continuation method. At each optimizer step it samples `K` candidate batches, computes the true LoRA-gradient cosine utility for each candidate against a utility-validation gradient, selects the highest-utility candidate by default, and then performs one normal SFT update on that batch. If this oracle does not beat `random_sft`, the gradient-cosine utility signal itself is not useful in the current setup.

`src/diagnose_evaluator.py` measures whether the learned evaluator can imitate true gradient utility inside the actual `K`-candidate selection problem. It writes:

- `outputs/evaluator_imitation.json`
- `outputs/evaluator_imitation.csv`

`src/check_gradient_signal.py` temporarily trains on candidate batches from the warmup checkpoint, restores the adapter after each temporary step, and reports whether gradient cosine correlates with validation-loss decrease. It writes:

- `outputs/gradient_signal_check.json`
- `outputs/gradient_signal_check.csv`

`outputs/diagnostic_summary.txt` prints the key pass/fail checks at the end of `run_all`.

## Noisy Pool Stress

Set `data.enable_synthetic_noise: true` or use `smoke_noise` / `full_noise` to corrupt only `D_train_pool`. Noise types include shuffled responses, truncated responses, generic responses, and verbose distractors. The processor stores `is_synthetic_noise` and `noise_type` for analysis, but does not feed the noise flag to the evaluator unless `data.feed_noise_feature_to_evaluator: true`.

## Repeated Seeds

Run multiple seeds with:

```bash
SEEDS="1 2 3" bash scripts/run_seeds.sh
```

If `SEEDS` is unset, `scripts/run_seeds.sh` reads the `seeds:` list from `configs/default.yaml` or from the config named by `CONFIG`.

Artifacts are written under `outputs/seeds/seed_<N>/`, then aggregated to:

- `outputs/seeds/seed_summary.json`
- `outputs/seeds/seed_results.csv`
- `outputs/seeds/seed_curves.csv`
- `outputs/seeds/seed_overhead.csv`
- `outputs/seeds/seed_selection_logs.csv`
- `outputs/seeds/seed_evaluator_diagnostics.csv`
- `outputs/seeds/seed_gradient_signal_summary.csv`
- `outputs/seeds/plots/seed_validation_loss_mean_std.png`
- `outputs/seeds/plots/seed_validation_loss_by_method_mean_std.png`
- `outputs/seeds/plots/seed_test_loss_mean_std.png`
- `outputs/seeds/plots/seed_validation_auc_mean_std.png`
- `outputs/seeds/plots/seed_selection_overhead_mean_std.png`
- `outputs/seeds/plots/seed_*_mean_std.png`
- `outputs/seeds/plots/seed_evaluator_*png`
- `outputs/seeds/plots/seed_gradient_signal_*png`

The seed curve CSV includes mean, standard deviation, and standard error at each validation step. Seed curve and trace plots show mean curves with a +/- 1 standard deviation band across seeds; seed scatter/histogram plots combine points or distributions from all seeds.

## Outputs

Important outputs include:

- `outputs/results.json`
- `outputs/results.csv`
- `outputs/curves/*.jsonl`
- `outputs/logs/*_overhead.json`
- `outputs/logs/adaptive_utility_sft_selected_batches.jsonl`
- `outputs/logs/oracle_gradient_sft_selected_batches.jsonl`
- `outputs/evaluator_metrics.json`
- `outputs/evaluator_dev_predictions.csv`
- `outputs/evaluator_imitation.json`
- `outputs/gradient_signal_check.json`
- `outputs/diagnostic_summary.txt`
- `outputs/plots/*.png`

Overhead logs track optimizer steps, candidate batches scored, gradient calls, and wall-clock seconds.

## GPU Expectations

The full run is intended for a single CUDA GPU with enough memory for Qwen2.5-0.5B LoRA SFT at sequence length 1024 and small micro-batches. The code uses bf16 when available, fp16 on CUDA otherwise, and fp32 on CPU. CPU smoke mode may be very slow because it still uses the real model.

On a 4x A100 node, prefer `scripts/run_all_multi_gpu.sh`. Shared model-heavy setup uses single-process `DataParallel` when multiple GPUs are selected. Continuation training is process-level parallelism across independent methods, not DDP, so it preserves the batch-level selection behavior and keeps method comparisons simple.

## Interpreting Results

The main comparison is final test SFT loss and perplexity across methods under the same number of continuation optimizer steps. The adaptive claim is stronger if:

- `oracle_gradient_sft` beats `random_sft`
- `adaptive_utility_sft` beats `random_sft`
- it is competitive with or better than `static_gradient_sft`
- `adaptive_utility_sft` beats `adaptive_shuffled_scores`
- selected-batch loss, length, or predicted utility changes over training

`outputs/evaluator_metrics.json` is the first sanity check. If dev ranking accuracy and Spearman are close to random, the adaptive run should be treated as a failed selector rather than a failed training loop. By default, `adaptive_utility_sft` still uses the evaluator even when it is weak, so the comparison against `adaptive_shuffled_scores` remains a direct diagnostic of whether evaluator predictions matter.

`outputs/evaluator_imitation.json` is the direct selection diagnostic. If top-1 agreement is near `1/K`, pairwise ranking accuracy is near 50%, and Spearman is near zero, the evaluator is not learning useful selection.

`outputs/gradient_signal_check.json` is the direct utility-label diagnostic. Positive correlation means gradient cosine points in the right direction; negative correlation suggests a sign or setup bug; near-zero correlation means the label is too noisy for this setup.

Adaptive logs include raw score std, standardized score std, selection entropy, selected probability, selected rank, and score gap. If entropy is near uniform and selected rank is random, adaptive selection is effectively random.

## Limitations

This is a first working version. Utility labels are approximate, evaluator features are deliberately cheap, and online adaptive selection is batch-level only. Gradient-alignment labels and static-gradient filtering are expensive by design, so their overhead is logged separately from optimizer steps.
