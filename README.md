---
title: Check Git
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Check Git Repos without cloning if its real LLM or fake
license: mit
---


# Check Git - ML Repo Analyzer

Quick tool to detect fake ML training repositories that are just API wrappers. For the community ["Wall of Shames"](https://github.com/wall-of-shames?view_as=public) Please read and learn and save the world from scammers with der Wannabe AI tools! Thanks from JADE, NCF, RustySafe and Friends

## What it does

Analyzes GitHub repositories to determine if they contain real machine learning training code or just wrapper scripts calling external APIs (OpenAI, Anthropic, etc).

## Features

- Pattern matching analysis (no token required)
- Optional LLM-powered deep analysis using HuggingFace Inference API
- Checks for real training indicators: torch.optim, loss.backward(), custom models
- Detects API wrapper patterns: openai.api, requests.post to APIs
- No data storage, runs entirely client-side


## Usage
[Demo](https://huggingface.co/spaces/Alibrown/Check-Git/)
> Enter a GitHub repository URL and click Analyze.

For LLM analysis, provide your HuggingFace token (free at https://huggingface.co/settings/tokens).

## How it works

Without token: Uses regex pattern matching to detect training loops and API calls.

With token: Uses Qwen2.5-Coder-32B on HuggingFace's free inference API for deeper code understanding.

## License

MIT + ESOL v 1.0

By VolkanSah aka AliBrown@HF for JADE and WoS
