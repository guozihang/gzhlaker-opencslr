# Agent Harness for OpenCSLR

**Version**: 1.0.0  
**Status**: Tier 1 Ready, Tier 2 Planned  
**Model**: DeepSeek (Chat + Reasoner)

## Overview

The Agent Harness is an experimental automation layer that uses LLM-driven agents to:
1. **Tier 1** (Zero GPU): Search decoder hyperparameters on trained checkpoints
2. **Tier 2** (Bounded orchestration): Generate and execute experiment matrices

This is **NOT** about inventing new models. It's about systematically exploring the configuration space defined by OpenCSLR's unified framework.

### Design Principles

1. **Bounded and transparent**: All actions are validated before execution
2. **Budget-aware**: Hard limits on GPU hours and disk space
3. **Deterministic**: All experiments use fixed seeds from whitelist
4. **Non-invasive**: Harness never modifies `core/` code
5. **Dev-only evaluation**: Agents search on dev set; test set run only once per config

---

## Architecture

```
agent/
├── README.md                # This file
├── env.py                   # Wraps core/main.py as an environment
├── deepseek_client.py       # OpenAI-compatible DeepSeek API client
├── validator.py             # Action schema validation + safety checks
├── search.py                # AIDE-style tree search for Tier 1
├── memory.py                # Experiment memory: config hash → results
├── budget.py                # GPU·day budget + disk/memory guards
└── examples/
    ├── tier1_decode_search.py   # Example: decoder hyperparameter search
    └── tier2_matrix_gen.py      # Example: experiment matrix generation
```

---

## Tier 1: Decoder Hyperparameter Search

**Goal**: Find optimal decoder settings (beam size, LM weight) for trained models.

**Input**: Trained checkpoint (e.g., `work_dir/vac_phoenix14/best_model.pt`)

**Output**: 
- Best decoder config (e.g., `beam_size=8, lm_weight=0.6`)
- WER improvement curve
- API cost and wall-clock time

**Why Zero GPU**: Decoding is CPU-bound and takes minutes, not hours.

### Action Protocol (Tier 1)

```json
{
  "action": "evaluate_decoder",
  "checkpoint": "work_dir/vac_phoenix14/best_model.pt",
  "dataset": "phoenix2014",
  "split": "dev",
  "decoder_config": {
    "decode_mode": "beam",
    "beam_size": 8,
    "lm_weight": 0.6
  },
  "hypothesis": "Increasing beam size to 8 may reduce WER by exploring more candidates"
}
```

**Validation checks**:
- Checkpoint file exists
- Decoder config keys are valid (whitelisted)
- `split` must be "dev" (test only allowed once per config)
- Seed is fixed (no random search)

**Execution**:
```bash
# Harness constructs and runs:
python core/main.py \
    --config <inferred_from_checkpoint> \
    --phase test \
    --load-weights work_dir/vac_phoenix14/best_model.pt \
    --dataset phoenix2014 \
    --split dev \
    --decoder-args '{"decode_mode": "beam", "beam_size": 8, "lm_weight": 0.6}'
```

**Result**:
```json
{
  "action_id": "decode_001",
  "wer": 19.2,
  "improvement": -0.5,  // vs. baseline greedy
  "time_seconds": 143,
  "api_tokens": 1200,
  "status": "success"
}
```

---

## Tier 2: Experiment Matrix Orchestration

**Goal**: Generate component replacement experiments (RQ1) and submit to training queue.

**Scope**: 
- Generate configs programmatically
- Submit via `script/run.sh` or watchdog
- Monitor logs and extract WER
- Do NOT auto-retry or debug training failures

**Example Task**: "Compare temporal modules: BiLSTM vs. TemporalConv vs. Transformer on Phoenix2014"

**Action Protocol (Tier 2)**:

```json
{
  "action": "propose_experiment_matrix",
  "base_config": "configs/unified_phoenix2014.yaml",
  "matrix": {
    "temporal_module": ["bilstm", "temporalconv", "transformer"]
  },
  "expected_gpu_hours": 9.0,  // 3 models × 3 GPU·hours each
  "rationale": "RQ1: Compare temporal modules under unified protocol"
}
```

**Validation checks**:
- All component names exist in registries
- Total GPU hours ≤ remaining budget
- No duplicate experiments (check memory)
- Seed is fixed across all configs

**Execution**:
- Generate 3 config files
- Submit to training queue (via run.sh)
- Monitor work_dir logs
- Parse WER from result.json

**Guard rails**:
- Agent cannot spawn training directly (must go through queue)
- Agent cannot modify submitted configs after queue entry
- Queue has its own GPU limit enforcement

---

## Security & Safety

### What Goes to the API

**Allowed** (sent to DeepSeek):
- Config dicts (YAML structure)
- Logs (training loss, WER numbers)
- CTM text (decoded glosses)
- Error messages

**Never sent**:
- Video pixel data
- Model weights
- Raw dataset files

### Hallucination Prevention

1. **Schema validation**: All actions validated against JSON schema before execution
2. **Registry checks**: Component names must exist in `core/registry.py`
3. **Seed whitelist**: Only approved seeds (e.g., 0, 1, 42) allowed
4. **Budget guard**: Disk/GPU/memory checked before every action
5. **Dry-run mode**: All experiments can be previewed without execution

### Example: Invalid Action Rejection

```python
# Agent proposes:
{"action": "train_model", "model": "gpt4_for_cslr", "seed": "random"}

# Validator rejects:
{
  "status": "invalid",
  "errors": [
    "Model 'gpt4_for_cslr' not in MODEL_REGISTRY",
    "Seed must be integer from whitelist [0, 1, 42], got 'random'"
  ]
}
# Agent receives rejection and retries with valid config
```

---

## DeepSeek Integration

### Model Selection

- **deepseek-chat** (V3.x): Structured action generation (JSON output)
- **deepseek-reasoner** (R1): Crash log diagnosis and strategy reflection (low frequency)

### Why DeepSeek?

1. **Cost-effective**: ~10× cheaper than GPT-4 for equivalent quality
2. **Strong reasoning**: R1 comparable to o1 on ML-Master benchmark
3. **OpenAI-compatible API**: Easy to swap models if needed
4. **Open-source precedent**: ML-Master paper (DeepSeek-R1, 29.33% on MLE-bench)

### API Configuration

```python
# deepseek_client.py
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Enable prompt caching for repeated system prompts
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[...],
    response_format={"type": "json_object"},  # Force JSON
    temperature=0.0  # Deterministic for reproducibility
)
```

### Cost Tracking

Every API call is logged with:
- Token count (input + output)
- Cost (USD)
- Cumulative cost per experiment
- Cost per WER point improvement

**Budget alert**: Harness stops if cost exceeds $50 per experiment (configurable).

---

## Memory System

### Experiment Memory

```python
# memory.py
memory = {
    "config_hash": "a3f2b1c9",  # SHA256 of config dict
    "results": {
        "dev_wer": 19.5,
        "test_wer": 20.1,  # Only if test was run
        "status": "valid",
        "skip_rate": 0.02
    },
    "efficiency": {
        "params_m": 45.2,
        "gpu_hours": 2.8,
        "peak_memory_gb": 8.1
    },
    "api_cost": 0.12,
    "timestamp": "2024-09-10T10:30:00Z"
}
```

### Deduplication

Before proposing an experiment:
1. Hash the config (ignore random seed, work_dir)
2. Check if hash exists in memory
3. If exists and valid, return cached result (no re-run)

### Persistent Storage

```python
# memory.json structure
{
  "experiments": {
    "a3f2b1c9": { ... },
    "b4e1c2d8": { ... }
  },
  "metadata": {
    "total_experiments": 12,
    "total_gpu_hours": 34.5,
    "total_api_cost": 2.34
  }
}
```

---

## Budget Guard

### GPU Budget

```python
# budget.py
class BudgetGuard:
    def __init__(self, max_gpu_hours=38.0):
        self.max_gpu_hours = max_gpu_hours
        self.used_gpu_hours = 0.0
    
    def check_action(self, estimated_gpu_hours):
        if self.used_gpu_hours + estimated_gpu_hours > self.max_gpu_hours:
            raise BudgetExceededError(
                f"Action would exceed budget: "
                f"{self.used_gpu_hours + estimated_gpu_hours:.1f} / {self.max_gpu_hours:.1f}"
            )
    
    def record_usage(self, actual_gpu_hours):
        self.used_gpu_hours += actual_gpu_hours
```

### Resource Monitoring

Before each action:
```python
# Check disk space
free_gb = shutil.disk_usage("/").free / (1024**3)
if free_gb < 10:
    raise ResourceError("Disk space < 10GB")

# Check memory
free_mem_gb = psutil.virtual_memory().available / (1024**3)
if free_mem_gb < 4:
    raise ResourceError("Free memory < 4GB")
```

---

## Action Validator

### Schema Definition

```python
# validator.py
ACTION_SCHEMAS = {
    "evaluate_decoder": {
        "type": "object",
        "required": ["checkpoint", "dataset", "split", "decoder_config"],
        "properties": {
            "checkpoint": {"type": "string", "pattern": r".*\.pt$"},
            "dataset": {"enum": ["phoenix2014", "phoenix2014t", "csl-daily"]},
            "split": {"enum": ["dev", "test"]},
            "decoder_config": {
                "type": "object",
                "properties": {
                    "decode_mode": {"enum": ["greedy", "beam"]},
                    "beam_size": {"type": "integer", "minimum": 1, "maximum": 20},
                    "lm_weight": {"type": "number", "minimum": 0.0, "maximum": 2.0}
                }
            }
        }
    },
    # ... other action types
}
```

### Validation Pipeline

```python
def validate_action(action_dict):
    # 1. Schema validation (JSON structure)
    validate_json_schema(action_dict, ACTION_SCHEMAS[action_dict["action"]])
    
    # 2. Registry check (component names exist)
    if "model" in action_dict:
        if not MODEL_REGISTRY.is_registered(action_dict["model"]):
            raise ValidationError(f"Model not registered: {action_dict['model']}")
    
    # 3. File existence check
    if "checkpoint" in action_dict:
        if not Path(action_dict["checkpoint"]).exists():
            raise ValidationError(f"Checkpoint not found: {action_dict['checkpoint']}")
    
    # 4. Seed whitelist
    if "seed" in action_dict:
        if action_dict["seed"] not in [0, 1, 42]:
            raise ValidationError(f"Seed not in whitelist: {action_dict['seed']}")
    
    # 5. Budget check
    budget_guard.check_action(action_dict.get("estimated_gpu_hours", 0))
    
    return True
```

---

## Tier 1 Example: Decoder Search

```python
# examples/tier1_decode_search.py
from agent.env import OpenCSLREnv
from agent.deepseek_client import DeepSeekClient
from agent.search import TreeSearch
from agent.memory import ExperimentMemory

# Initialize
env = OpenCSLREnv()
client = DeepSeekClient(model="deepseek-chat")
memory = ExperimentMemory("agent/memory.json")
search = TreeSearch(env, client, memory)

# Run search
result = search.run(
    task="Find best decoder config for VAC checkpoint",
    checkpoint="work_dir/vac_phoenix14/best_model.pt",
    dataset="phoenix2014",
    split="dev",
    baseline_wer=19.9,
    max_iterations=10,
    budget_usd=5.0
)

print(f"Best config: {result['best_config']}")
print(f"Best WER: {result['best_wer']}")
print(f"Improvement: {result['improvement']}")
print(f"Total cost: ${result['total_cost']:.2f}")
```

**Expected output**:
```
Iteration 1: beam_size=5, lm_weight=0.0 → WER=19.7 (-0.2)
Iteration 2: beam_size=10, lm_weight=0.0 → WER=19.5 (-0.4)
Iteration 3: beam_size=10, lm_weight=0.3 → WER=19.3 (-0.6)
Iteration 4: beam_size=10, lm_weight=0.6 → WER=19.2 (-0.7) ✓ Best
Iteration 5: beam_size=10, lm_weight=0.9 → WER=19.4 (-0.5)

Best config: {"beam_size": 10, "lm_weight": 0.6}
Best WER: 19.2%
Improvement: -0.7% absolute
Total cost: $0.38
```

---

## Tier 2 Example: Matrix Generation

```python
# examples/tier2_matrix_gen.py
from agent.env import OpenCSLREnv
from agent.deepseek_client import DeepSeekClient
from agent.matrix import MatrixOrchestrator

env = OpenCSLREnv()
client = DeepSeekClient(model="deepseek-chat")
orchestrator = MatrixOrchestrator(env, client)

# Generate RQ1 experiment matrix
matrix = orchestrator.generate_matrix(
    research_question="RQ1: Compare temporal modules",
    base_config="configs/unified_phoenix2014.yaml",
    dimensions={
        "temporal_module": ["bilstm", "temporalconv", "transformer"]
    },
    gpu_budget=9.0
)

# Preview generated configs
print(f"Generated {len(matrix.configs)} configs")
for cfg in matrix.configs:
    print(f"  - {cfg['experiment_name']}: {cfg['temporal_module']}")

# Submit to queue (dry-run by default)
orchestrator.submit_matrix(matrix, dry_run=True)
```

---

## Validation Checklist

Before Tier 1 release:

- [ ] `env.py` wraps `core/main.py` correctly
- [ ] `validator.py` rejects invalid actions
- [ ] `memory.py` deduplicates experiments
- [ ] `budget.py` enforces GPU limits
- [ ] Decoder search completes on 1 checkpoint
- [ ] API cost tracking works
- [ ] All actions logged
- [ ] README documents action protocol
- [ ] No `core/` files modified

Before Tier 2 release:

- [ ] Matrix generation produces valid configs
- [ ] Queue submission works
- [ ] Log monitoring extracts WER
- [ ] GPU budget enforced across queue
- [ ] Agent handles training failures gracefully
- [ ] Cost per WER point calculated

---

## Paper Integration

### Where to Mention

**Section**: Reproducibility & Extensibility

**Content**:
> To demonstrate the framework's automation potential, we provide an optional agent-based harness (Tier 1) that searches decoder hyperparameters on trained checkpoints. Using DeepSeek-Chat with structured action validation, the harness found optimal beam search settings (beam_size=10, lm_weight=0.6) with 0.7% WER improvement at $0.38 API cost. This shows that unified interfaces enable programmatic exploration without modifying core code.

**Claim scope**:
- Harness demonstrates automation potential
- Cost-benefit analysis (API cost vs. manual tuning time)
- **NOT claiming**: "Agents beat human tuning" or "Autonomous training loops"

---

## FAQ

### Q: Will the agent modify `core/` code?
**A**: No. The harness only interacts via CLI (`python core/main.py`) and config files.

### Q: What if the agent generates invalid configs?
**A**: The validator rejects them before execution. Agent sees rejection and retries.

### Q: Can the agent train models autonomously?
**A**: Not in v1.0.0. Tier 2 generates configs and submits to queue, but doesn't debug failures or retry automatically.

### Q: What about overfitting to dev set?
**A**: Test set is run only ONCE per config. All agent search happens on dev. Test WER goes directly to paper table.

### Q: Can I use a different LLM?
**A**: Yes. `deepseek_client.py` uses OpenAI-compatible API, so you can swap to GPT-4, Claude, or local models.

---

## References

- **ML-Master**: DeepSeek-R1 on MLE-bench (29.33%, 12h) - [Paper](https://arxiv.org/abs/...)
- **AIDE**: Agent-based hypothesis exploration - [Paper](https://arxiv.org/abs/...)
- **DeepSeek API**: https://platform.deepseek.com/docs

---

**Last Updated**: 2024-09-09  
**Status**: Tier 1 design complete, implementation pending  
**Contact**: Open an issue for questions
