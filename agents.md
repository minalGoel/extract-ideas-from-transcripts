# LLM Backend Guide

Three interchangeable backends for calling Claude. All share the same interface:

```python
backend.complete(system: str, user: str) -> str
```

Select via CLI flag: `--backend azure | anthropic | bedrock`

---

## Azure AI Foundry (default)

Claude deployed on Azure. Uses the Anthropic SDK pointed at your Azure endpoint.

**Env vars:**
```bash
export AZURE_ANTHROPIC_ENDPOINT="https://<resource>-<id>-<region>.services.ai.azure.com/anthropic/"
export AZURE_ANTHROPIC_API_KEY="your-key"
export AZURE_ANTHROPIC_DEPLOYMENT="claude-opus-4-6"   # or whatever model you deployed
```

**Test:**
```bash
python extract_from_parquet.py --backend azure --test
```

**Notes:**
- The endpoint URL must include the trailing `/anthropic/` path
- The deployment name must match what's configured in your Azure AI Foundry resource
- Also accepts legacy env vars `AZURE_ENDPOINT`, `AZURE_API_KEY`, `AZURE_MODEL` as fallbacks

---

## Anthropic Direct

Simplest option — uses your Anthropic API key directly.

**Env vars:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Test:**
```bash
python extract_from_parquet.py --backend anthropic --test
```

**Notes:**
- Get a key at [console.anthropic.com](https://console.anthropic.com)
- Default model: `claude-opus-4-6` (override with `ANTHROPIC_MODEL` env var)
- Rate limits are lower than Bedrock/Azure — keep `MAX_WORKERS` at 3 or less

---

## AWS Bedrock

Claude via AWS. Uses boto3 credentials (key pair, IAM role, or SSO profile).

**Env vars:**
```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
```

Or use an IAM role / SSO profile (no env vars needed if configured in `~/.aws/`).

**Test:**
```bash
python extract_from_parquet.py --backend bedrock --test
```

**Notes:**
- Default model: `anthropic.claude-opus-4-5-20251101-v1:0` (override with `BEDROCK_MODEL` env var)
- Make sure Claude is enabled in your Bedrock console for the chosen region
- Bedrock model IDs differ from direct API — check your AWS console for exact model IDs

---

## Switching backends

The `--backend` flag selects which backend to use at runtime. No code changes needed:

```bash
# Azure (default if --backend omitted)
python extract_from_parquet.py --backend azure

# Anthropic
python extract_from_parquet.py --backend anthropic

# Bedrock
python extract_from_parquet.py --backend bedrock
```

The progress file (`extract_ideas_progress.jsonl`) is backend-agnostic — you can start with one backend, hit a rate limit, and resume with another. Already-processed videos are always skipped regardless of which backend you use.
