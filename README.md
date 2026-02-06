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

Quick tool to detect fake ML training repositories that are just API wrappers.

Built for the ["Wall of Shames"](https://github.com/wall-of-shames?view_as=public) community - help save the world from wannabe AI scammers.

## What it does

Analyzes GitHub repositories to determine if they contain real machine learning training code or just wrapper scripts calling external APIs (OpenAI, Anthropic, etc).

## Features

- Pattern matching analysis (no token required)
- Optional LLM-powered deep analysis using HuggingFace Inference API
- Checks for real training indicators: torch.optim, loss.backward(), custom models
- Detects API wrapper patterns: openai.api, requests.post to APIs
- No data storage, runs entirely client-side

## Usage

**Live Demo:** https://huggingface.co/spaces/Alibrown/Check-Git

Enter a GitHub repository URL and click Analyze.

For LLM-powered analysis, provide your HuggingFace token (free at https://huggingface.co/settings/tokens).

## How it works

**Without token:** Uses regex pattern matching to detect training loops and API calls.

**With token:** Uses Qwen2.5-Coder-32B on HuggingFace's free inference API for deeper code understanding.

## License

Dual-licensed under [MIT](LICENSE) + [ESOL v1.1](ESOL) 

---

**Built with ❤️ by VolkanSah (AliBrown@HF) for JADE, NCF, RustySafe and the Wall of Shames community**

*With assistance from Claude (Anthropic)*

