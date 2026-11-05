# kb-agent rag-eval integration

## Install the local package

From `kb-agent`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-eval-github.txt
# After PyPI publication:
.\.venv\Scripts\python.exe -m pip install "rag-eval>=0.1.0,<0.2.0"
```

This installs the same pinned GitHub Release version used by CI. The version
range in `requirements-eval.txt` is the post-PyPI release dependency.

## Convert the 50-question golden set

```powershell
.\.venv\Scripts\python.exe scripts\migrate_rag_eval_golden_set.py
```

The converter writes:

- `evals/rag_eval/kb_agent_retrieval_50`
- `evals/rag_eval/kb_agent_l3_50`

The original `evals/golden_set.jsonl` is not modified.

Validate:

```powershell
.\.venv\Scripts\rag-eval.exe validate evals\rag_eval\kb_agent_retrieval_50
.\.venv\Scripts\rag-eval.exe validate evals\rag_eval\kb_agent_l3_50
```

## Retrieval-only

```powershell
.\.venv\Scripts\rag-eval.exe run `
  --config configs\eval.kb_retrieval.yaml
```

`KbAgentVectorRetrievalAdapter` calls `rag_adapter.rag.retrieve()` directly in
FAISS vector mode with rerank disabled. It does not generate answers or call an
LLM. `KbAgentRetrievalAdapter` remains available for BM25, vector, or hybrid
configuration through environment variables.

Configuration:

- `KB_AGENT_EVAL_STRATEGY`
- `KB_AGENT_EVAL_MODE`
- `KB_AGENT_EVAL_RERANK`
- `KB_AGENT_EVAL_TOP_K`

## L3 agent

```powershell
.\.venv\Scripts\rag-eval.exe run `
  --config configs\eval.kb_agent.yaml
```

Generation and LLM-based grading are disabled by default:

- `KB_AGENT_EVAL_GENERATE=false`
- `KB_AGENT_EVAL_LLM_REWRITE=false`
- `KB_AGENT_EVAL_LLM_GRADER=false`

Enable them only for an intentional generated evaluation run. The adapter name
changes with generated/retrieval-only mode, so prediction caches do not cross
between the two.

Low-confidence interruptions are reported as `human_review`; the adapter does
not auto-approve or resume them.

The checked-in L3 configuration uses `KbAgentL3VectorAdapter` for a
deterministic FAISS baseline. `KbAgentL3Adapter` remains available for hybrid
and rerank runs through `KB_AGENT_EVAL_L3_MODE` and
`KB_AGENT_EVAL_L3_RERANK`.

## Answer judge

`KbAgentAnswerJudge` implements the rag-eval judge interface and reuses the
existing `evals/judge_answers.py` prompt and parser. The judge remains
advisory and is not part of the deterministic gate.

## Baselines

Store validated new-platform baselines under:

```text
evals/rag_eval/baselines/
```

Runtime predictions and reports are written under
`evals/reports/rag_eval/` and ignored by Git.

## CI

- PR CI runs unit tests, both dataset validators, and the rag-eval package
  contract/gate tests without Docker or Elasticsearch.
- Nightly CI fetches annual reports, rebuilds the local index when missing,
  runs the 50-question retrieval and L3 gates, and uploads reports.
- If `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` repository secrets are
  configured, nightly also runs one generated L3 pass and the advisory answer
  judge.
