# ============================================
# LICENSE & SOURCE
# ============================================
# Licensed under MIT + ESOL v1.1
# Source: https://github.com/VolkanSah/Check-Git-ML-Repo-Analyzer
# 
# What is Open Source? For scammers & newcomers:
# Learn what Open Source IS and what it IS NOT:
# https://github.com/Wall-of-Shames/What-is-Open-Source
# ============================================
import streamlit as st
import requests
import re
import os
import tempfile
from typing import Dict, List, Tuple
import json
from huggingface_hub import InferenceClient

# ============================================
# STREAMLIT PERMISSION HACK by VolkanSah :D
# ============================================
TEMP_STREAMLIT_HOME = os.path.join(tempfile.gettempdir(), "st_config_workaround")
os.makedirs(TEMP_STREAMLIT_HOME, exist_ok=True)
os.environ["STREAMLIT_HOME"] = TEMP_STREAMLIT_HOME
os.environ["STREAMLIT_GATHER_USAGE_STATS"] = "false"
CONFIG_PATH = os.path.join(TEMP_STREAMLIT_HOME, "config.toml")
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "w") as f:
        f.write("[browser]\ngatherUsageStats = false\n")

# ============================================
# LLM-POWERED ANALYZER
# ============================================

class MLRepoAnalyzerLLM:
    def __init__(self, hf_token: str = None):
        self.hf_token = hf_token
        if hf_token:
            self.client = InferenceClient(token=hf_token)
        
        # Fallback patterns (wenn kein Token)
        self.fake_indicators = [
            r'openai\.', r'anthropic\.', r'cohere\.', 
            r'replicate\.', r'api\.mistral', r'groq\.',
            r'requests\.post.*api', r'urllib.*api'
        ]
        self.legit_indicators = [
            r'torch\.optim', r'loss\.backward\(\)', r'model\.train\(\)',
            r'optimizer\.step\(\)', r'tf\.keras\.optimizers',
            r'from\s+transformers\s+import\s+Trainer',
            r'accelerator\.backward', r'DeepSpeed',
            r'torch\.nn\.Module', r'forward\(self'
        ]
    
    def extract_repo_info(self, url: str) -> Tuple[str, str, str]:
        """Extract owner, repo, branch from GitHub URL"""
        pattern = r'github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+))?'
        match = re.search(pattern, url)
        if not match:
            raise ValueError("Invalid GitHub URL")
        owner, repo = match.group(1), match.group(2)
        branch = match.group(3) or 'main'
        return owner, repo.replace('.git', ''), branch
    
    def fetch_repo_tree(self, owner: str, repo: str, branch: str) -> List[Dict]:
        """Fetch file tree via GitHub API"""
        api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        response = requests.get(api_url, timeout=10)
        if response.status_code != 200:
            raise Exception(f"GitHub API error: {response.status_code}")
        return response.json().get('tree', [])
    
    def fetch_file_content(self, owner: str, repo: str, branch: str, path: str) -> str:
        """Fetch raw file content"""
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        response = requests.get(raw_url, timeout=10)
        return response.text if response.status_code == 200 else ""
    
    def analyze_with_llm(self, code_snippet: str, filename: str) -> Dict:
        """Use HF Inference API to analyze code"""
        if not self.hf_token:
            return None
        
        prompt = f"""Analyze this Python file from a machine learning repository: {filename}

Code snippet:
```python
{code_snippet[:2000]}  # Limit to avoid token limits
```

Determine if this is:
1. REAL ML TRAINING CODE (contains actual model training, backprop, optimizers)
2. API WRAPPER (just calls external APIs like OpenAI, Anthropic, etc.)
3. UNCLEAR

Respond in JSON format:
{{
  "classification": "REAL_TRAINING|API_WRAPPER|UNCLEAR",
  "confidence": 0-100,
  "reasoning": "brief explanation",
  "key_indicators": ["indicator1", "indicator2"]
}}"""

        try:
            # Use Qwen2.5-Coder or similar code-focused model
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model="Qwen/Qwen2.5-Coder-32B-Instruct",  # Free on HF Inference
                max_tokens=500,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content
            
            # Extract JSON (handle markdown code blocks)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                # Try direct parse
                return json.loads(result_text)
                
        except Exception as e:
            st.warning(f"LLM analysis failed for {filename}: {e}")
            return None
    
    def analyze_file_structure(self, files: List[Dict]) -> Dict:
        """Quick structure check"""
        py_files = [f for f in files if f['path'].endswith('.py')]
        
        return {
            'has_train_script': any('train' in f['path'].lower() for f in py_files),
            'has_model_files': any('model' in f['path'].lower() for f in py_files),
            'has_config': any(f['path'].endswith(('.yaml', '.yml', '.json', '.toml')) for f in files),
            'has_requirements': any('requirements' in f['path'] or 'pyproject.toml' in f['path'] for f in files),
            'python_file_count': len(py_files)
        }
    
    def analyze_with_patterns(self, content: str) -> Tuple[int, int]:
        """Fallback pattern matching"""
        fake_score = sum(5 for pattern in self.fake_indicators if re.search(pattern, content, re.IGNORECASE))
        legit_score = sum(10 for pattern in self.legit_indicators if re.search(pattern, content, re.IGNORECASE))
        return fake_score, legit_score
    
    def classify_repo(self, url: str, use_llm: bool = True) -> Dict:
        """Main classification"""
        try:
            owner, repo, branch = self.extract_repo_info(url)
            files = self.fetch_repo_tree(owner, repo, branch)
            
            structure = self.analyze_file_structure(files)
            py_files = [f for f in files if f['path'].endswith('.py')][:10]
            
            llm_results = []
            pattern_fake_score = 0
            pattern_legit_score = 0
            
            for file_info in py_files:
                content = self.fetch_file_content(owner, repo, branch, file_info['path'])
                if not content:
                    continue
                
                # LLM Analysis (if token available)
                if use_llm and self.hf_token:
                    llm_result = self.analyze_with_llm(content, file_info['path'])
                    if llm_result:
                        llm_results.append({
                            'file': file_info['path'],
                            'result': llm_result
                        })
                
                # Pattern fallback
                fake, legit = self.analyze_with_patterns(content)
                pattern_fake_score += fake
                pattern_legit_score += legit
            
            # Combine LLM + Pattern results
            if llm_results:
                llm_real_count = sum(1 for r in llm_results if r['result']['classification'] == 'REAL_TRAINING')
                llm_fake_count = sum(1 for r in llm_results if r['result']['classification'] == 'API_WRAPPER')
                
                # LLM gets more weight
                total_score = (llm_real_count * 30) - (llm_fake_count * 30) + (pattern_legit_score - pattern_fake_score)
            else:
                total_score = pattern_legit_score - pattern_fake_score
            
            # Verdict
            if total_score > 30:
                verdict = "✅ LEGIT - Real ML Training Code"
                confidence = "High"
            elif total_score > 0:
                verdict = "⚠️ MIXED - Contains some training code"
                confidence = "Medium"
            else:
                verdict = "❌ FAKE - API Wrapper / No Real Training"
                confidence = "High"
            
            return {
                'verdict': verdict,
                'confidence': confidence,
                'score': total_score,
                'structure': structure,
                'llm_results': llm_results,
                'pattern_scores': {
                    'fake': pattern_fake_score,
                    'legit': pattern_legit_score
                },
                'repo_info': f"{owner}/{repo}@{branch}"
            }
            
        except Exception as e:
            return {'error': str(e)}

# ============================================
# STREAMLIT UI
# ============================================

st.set_page_config(page_title="ML Repo Detector 🔍", page_icon="🤖", layout="wide")

st.title("🤖 ML Training Repo Analyzer (LLM-Powered)")
st.markdown("**AI-powered detection of fake ML repos using your HuggingFace token**")

# Token input in sidebar
with st.sidebar:
    st.markdown("### 🔑 HuggingFace Setup")
    hf_token = st.text_input(
        "HF Token (optional)",
        type="password",
        help="Get your free token at https://huggingface.co/settings/tokens"
    )
    
    use_llm = st.checkbox(
        "Use LLM Analysis",
        value=bool(hf_token),
        disabled=not hf_token,
        help="Requires HF token. Uses Qwen2.5-Coder for deep analysis"
    )
    
    st.markdown("---")
    st.markdown("### 🛠️ Models Used")
    if use_llm:
        st.success("✅ Qwen2.5-Coder-32B (Free)")
    else:
        st.info("📊 Pattern Matching Only")
    
    st.markdown("---")
    st.markdown("### 💡 How it works")
    st.markdown("""
    **With LLM:**
    - Deep code understanding
    - Context-aware analysis
    - Higher accuracy
    
    **Without LLM:**
    - Pattern matching
    - Regex-based detection
    - Still pretty good!
    """)

# Main interface
analyzer = MLRepoAnalyzerLLM(hf_token=hf_token if hf_token else None)

repo_url = st.text_input(
    "GitHub Repository URL",
    placeholder="https://github.com/username/repo",
    help="Enter a public GitHub repository URL"
)

col1, col2 = st.columns([1, 4])
with col1:
    analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)

if analyze_btn:
    if not repo_url:
        st.error("Enter a GitHub URL!")
    else:
        with st.spinner("🔍 Analyzing repository..." + (" (using LLM)" if use_llm else " (pattern matching)")):
            result = analyzer.classify_repo(repo_url, use_llm=use_llm and bool(hf_token))
            
            if 'error' in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                # Verdict
                st.markdown("---")
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"## {result['verdict']}")
                with col2:
                    st.metric("Confidence", result['confidence'])
                with col3:
                    st.metric("Score", result['score'])
                
                # LLM Results
                if result.get('llm_results'):
                    st.markdown("### 🤖 LLM Analysis Results")
                    for llm_res in result['llm_results'][:5]:
                        with st.expander(f"📄 {llm_res['file']}"):
                            res = llm_res['result']
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                classification = res.get('classification', 'UNKNOWN')
                                if classification == 'REAL_TRAINING':
                                    st.success(f"✅ {classification}")
                                elif classification == 'API_WRAPPER':
                                    st.error(f"❌ {classification}")
                                else:
                                    st.warning(f"⚠️ {classification}")
                            
                            with col2:
                                st.metric("Confidence", f"{res.get('confidence', 0)}%")
                            
                            st.markdown(f"**Reasoning:** {res.get('reasoning', 'N/A')}")
                            
                            if res.get('key_indicators'):
                                st.markdown("**Key Indicators:**")
                                for indicator in res['key_indicators']:
                                    st.markdown(f"- {indicator}")
                
                # Pattern Analysis (fallback/additional)
                st.markdown("### 📊 Pattern Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Legit Patterns", result['pattern_scores']['legit'])
                with col2:
                    st.metric("Fake Patterns", result['pattern_scores']['fake'])
                
                # Structure
                st.markdown("### 📁 Repository Structure")
                struct = result['structure']
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Python Files", struct['python_file_count'])
                with cols[1]:
                    st.write("✅" if struct['has_train_script'] else "❌", "train.py")
                with cols[2]:
                    st.write("✅" if struct['has_model_files'] else "❌", "model files")
                with cols[3]:
                    st.write("✅" if struct['has_config'] else "❌", "configs")

# Footer
st.markdown("---")
st.markdown("**💡 Your HF token = your quota. No data stored. Analysis runs on HF's free inference API.**")
