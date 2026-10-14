<div align="center">

<img alt="JARVIS AI Hero Banner" src="assets/jarvis_hero_banner.png" width="100%" style="border-radius: 12px; box-shadow: 0 8px 30px rgba(0, 240, 255, 0.15);" />

# ⚡ JARVIS: Autonomous Personal AI Assistant
### *Private On-Device Intelligence &bull; Exclusive Real-Time AI News via The Deep View*

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Local LLM Engine](https://img.shields.io/badge/Ollama-Local%20Inference-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![Model](https://img.shields.io/badge/Model-Gemma%204-FF6F00?style=for-the-badge&logo=google&logoColor=white)](https://ollama.com/library/gemma4)
[![Context Window](https://img.shields.io/badge/Context-16K%20(16%2C384)-6f42c1?style=for-the-badge)](https://github.com/Xeneium/Jarvis)
[![Web Research](https://img.shields.io/badge/News%20Source-The%20Deep%20View%20Only-00f0ff?style=for-the-badge&logo=rss&logoColor=black)](https://www.thedeepview.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-2ea44f?style=for-the-badge)](LICENSE)

<p align="center">
  <a href="#-key-features"><b>Features</b></a> •
  <a href="#-system-architecture"><b>Architecture</b></a> •
  <a href="#-quick-start"><b>Quick Start</b></a> •
  <a href="#-the-deep-view-integration"><b>Deep View Engine</b></a> •
  <a href="#-configuration"><b>Configuration</b></a> •
  <a href="#-security--privacy"><b>Security</b></a> •
  <a href="#-license"><b>License</b></a>
</p>

---

</div>

## 🌟 Executive Overview

**JARVIS** is a production-grade, local-first autonomous personal AI assistant engineered for privacy, speed, and zero hallucination in real-time AI developments. Built to run entirely on personal hardware, Jarvis decouples everyday reasoning and web retrieval from monolithic, closed-source cloud providers.

Unlike traditional AI wrappers that indiscriminately search the web—pulling in SEO spam, advertisement farms, and unverified blogs—**Jarvis restricts all web research exclusively to [The Deep View](https://www.thedeepview.com/)**, the premier authoritative curation of artificial intelligence breakthroughs, industry benchmarks, and frontier developments.

```
       ┌────────────────────────────────────────────────────────┐
       │                  SINGLE COMMAND LAUNCH                 │
       │                   $ python main.py                     │
       └───────────────────────────┬────────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   ┌───────────────────────┐               ┌──────────────────────────┐
   │  100% Local Inference  │               │    Verified Intelligence │
   │   Ollama / Gemma 4    │               │ Exclusive The Deep View  │
   │      16K Context      │               │   Real-Time Newsletter   │
   └───────────────────────┘               └──────────────────────────┘
```

---

## 🚀 Key Features

| Capability | Specification | Architectural Value |
| :--- | :--- | :--- |
| **🧠 Local LLM Execution** | `Ollama` (`gemma4:latest`) | Zero data leakage; fully offline reasoning with zero cloud telemetry. |
| **⚡ 16K Optimized Context** | `16,384 tokens` | High-efficiency attention window tuned for deep multi-turn ReAct cycles and prompt responsiveness. |
| **🌐 Exclusive Retrieval** | `thedeepview.com` Only | Pinned domain whitelist. No fallback to Google, Bing, or generic web indices. |
| **🔄 Native ReAct Engine** | Thought-Action-Observation | Dynamic tool calling with multi-hop retrieval and contextual self-correction. |
| **🛡️ SSRF & Network Guard** | DNS Pinning & URL Parsing | Strict defense-in-depth preventing local host probing or unverified URL traversal. |
| **💻 Zero-Config UX** | `python main.py` | Automatic Ollama connectivity check, model fallback protection, and immediate `You>` prompt. |

---

## 🏗️ System Architecture

Jarvis employs a clean, decoupled 5-tier architecture that cleanly separates user interaction, model reasoning, agentic tool dispatch, and network-bounded retrieval.

### 3D Layered Architecture

```mermaid
flowchart TD
    %% Global Styling
    classDef clientLayer fill:#161b22,stroke:#58a6ff,stroke-width:2px,color:#f0f6fc;
    classDef agentLayer fill:#1f242c,stroke:#a371f7,stroke-width:2px,color:#f0f6fc;
    classDef modelLayer fill:#161b22,stroke:#ff7b72,stroke-width:2px,color:#f0f6fc;
    classDef toolsLayer fill:#1f242c,stroke:#3fb950,stroke-width:2px,color:#f0f6fc;
    classDef retrievalLayer fill:#0d1117,stroke:#00f0ff,stroke-width:3px,color:#f0f6fc;

    subgraph CLIENT[" 💻 Tier 1: User Interface & Interaction "]
        CLI["🖥️ Interactive CLI\n(python main.py)"]
        PROMPT["📝 Clean 'You>' Console Interface"]
    end

    subgraph ORCHESTRATION[" ⚙️ Tier 2: Cognitive ReAct Loop "]
        AGENT["🧠 Native ReAct Agent\n(OpenJarvis Engine)"]
        CTX["📦 Context Governor\n(Strict 16,384 Window)"]
        PARSER["🔍 Tool Call Dispatcher\n& Observation Synthesizer"]
    end

    subgraph MODEL[" 🤖 Tier 3: Local Inference Engine "]
        OLLAMA["🦙 Ollama Runtime Daemon\n(http://localhost:11434)"]
        GEMMA["✨ Gemma 4:latest\n(Fast High-Fidelity Reasoning)"]
    end

    subgraph TOOLS[" 🛠️ Tier 4: Agent Tool Registry "]
        REG["📚 Integrated Tools\n(CodeAct, Filesystem, Calculator)"]
        DEEP_TOOL["🛰️ The Deep View Research Tool\n(Registered Action: search_deep_view)"]
    end

    subgraph RETRIEVAL[" 🌐 Tier 5: Exclusive Verified Web Intelligence "]
        GUARD["🛡️ Domain Whitelist & SSRF Guard\n(Strict: *.thedeepview.com)"]
        SITEMAP["🗺️ XML Sitemap & Archive Crawler\n(Live Post Discovery)"]
        PARSER_HTML["📰 Article Content Extractor\n(Title, Date, Summary, Key Takeaways)"]
        SOURCE[("🌐 The Deep View\nhttps://www.thedeepview.com/")]
    end

    %% Connections
    CLI --> PROMPT
    PROMPT --> AGENT
    AGENT <--> CTX
    AGENT <--> PARSER
    AGENT <--> OLLAMA
    OLLAMA <--> GEMMA

    PARSER --> REG
    PARSER --> DEEP_TOOL
    DEEP_TOOL --> GUARD
    GUARD --> SITEMAP
    SITEMAP --> SOURCE
    SOURCE --> PARSER_HTML
    PARSER_HTML --> DEEP_TOOL
    DEEP_TOOL --> PARSER
    AGENT --> CLI

    class CLI,PROMPT clientLayer;
    class AGENT,CTX,PARSER agentLayer;
    class OLLAMA,GEMMA modelLayer;
    class REG,DEEP_TOOL toolsLayer;
    class GUARD,SITEMAP,PARSER_HTML,SOURCE retrievalLayer;
```

---

### Request Execution Sequence

When you ask a question requiring contemporary AI knowledge, Jarvis dynamically triggers an iterative ReAct retrieval cycle:

```mermaid
sequenceDiagram
    autonumber
    actor User as 👤 You
    participant CLI as 🖥️ main.py
    participant Agent as 🧠 ReAct Agent
    participant LLM as 🦙 Ollama (Gemma 4:latest)
    participant DeepView as 🛰️ The Deep View Tool
    participant Web as 🌐 https://www.thedeepview.com/

    User->>CLI: "What are the latest updates on AI recently?"
    CLI->>Agent: Send user query (Prompt with 16K Context)
    Agent->>LLM: Formulate thought and check tool requirements
    LLM-->>Agent: Action: search_deep_view(query="latest updates on AI")
    Agent->>DeepView: Execute search_deep_view()
    Note over DeepView,Web: Enforce strict domain filter (*.thedeepview.com)
    DeepView->>Web: Fetch sitemap.xml & latest newsletter articles
    Web-->>DeepView: Return live headlines, summaries, and key points
    DeepView-->>Agent: Observation: Structured Deep View briefings
    Agent->>LLM: Ingest real-time verified data into 16K context
    LLM-->>Agent: Synthesize fact-anchored response with citations
    Agent->>CLI: Stream final answer to terminal
    CLI->>User: ⚡ Verified AI updates with The Deep View sources!
```

---

## ⚡ Quick Start

### 1. Prerequisites
- **Operating System:** Windows, macOS, or Linux
- **Python:** `3.10`, `3.11`, or `3.12`
- **Ollama:** Installed and running locally
  ```bash
  # Ensure Ollama has pulled gemma4
  ollama pull gemma4:latest
  ```

### 2. Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/Xeneium/Jarvis.git
cd Jarvis

# 2. Create and activate a Python virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -e .
```

### 3. Launch Jarvis

Run the single master command:

```bash
python main.py
```

You will immediately see the interactive console:

```text
================================================================================
  JARVIS: Autonomous Personal AI Assistant
  Model: gemma4:latest (Ollama) | Context: 16,384 tokens
  Exclusive Web Research: The Deep View (https://www.thedeepview.com/)
================================================================================

Type 'exit', 'quit', or 'q' to end the session.

You> What are the latest developments in AI foundation models?
Jarvis is thinking...
[Tool] Querying The Deep View archives...
[Tool] Retrieved latest newsletters from thedeepview.com
Jarvis> According to recent editions of The Deep View, the major focus has been...
```

---

## 🔬 The Deep View Integration

Jarvis is configured with an exclusive, tamper-proof retrieval mechanism that prevents general search engine fallbacks:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        EXCLUSIVE RETRIEVAL RULES                       │
│                                                                        │
│  ✅  Permitted:   https://www.thedeepview.com/                         │
│  ✅  Permitted:   https://*.thedeepview.com/newsletters/*              │
│                                                                        │
│  ❌  BLOCKED:     Google, Bing, DuckDuckGo, Tavily, SerpAPI            │
│  ❌  BLOCKED:     Twitter / X, Reddit, Medium, Substack                │
│  ❌  BLOCKED:     Arbitrary external blogs & unvetted domains          │
└────────────────────────────────────────────────────────────────────────┘
```

### Retrieval Capabilities
1. **Dynamic Sitemap Discovery:** Automatically inspects The Deep View's live XML sitemaps to locate new issues within seconds of publication.
2. **Key Takeaway Parser:** Extracts core breakthroughs, model releases, benchmark scores, and executive summaries while stripping ads and tracker scripts.
3. **Transparent Citations:** Every real-time response provides article dates, headlines, and original URLs from `thedeepview.com`.

---

## ⚙️ Configuration & Environment

Configuration is governed by environment variables and sensible production defaults. Copy the template to customize:

```bash
cp .env.example .env
```

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Endpoint for the local Ollama daemon. |
| `OLLAMA_MODEL` | `gemma4:latest` | Target local model. Preserved by default. |
| `OLLAMA_NUM_CTX` | `16384` | Sets context window length to 16K tokens. |
| `DEEP_VIEW_MAX_RESULTS` | `5` | Maximum number of newsletter articles fetched per search. |
| `DEEP_VIEW_REQUEST_TIMEOUT` | `15` | Request timeout in seconds for deepview queries. |

---

## 🛡️ Security & Privacy Architecture

Jarvis is designed from first principles with a zero-trust local-first security posture:

- **Strict Network Boundary:** The agent cannot initiate arbitrary outbound connections to external search engines or untrusted domains.
- **SSRF Protection:** URL validation enforces strict host checking against `thedeepview.com`, preventing intranet or loopback exploitation.
- **Zero Cloud Telemetry:** Inference runs 100% on localhost via Ollama. No user prompts, chat histories, or trace logs are transmitted to external servers.
- **Clean Repo History:** Sanitized of binary bloat and foreign commit histories.

---

## 📂 Repository Structure

```text
Jarvis/
├── main.py                     # Single-command CLI entrypoint with ReAct orchestration
├── pyproject.toml              # Project metadata & dependency specifications
├── .gitignore                  # Production Git ignore rules (binaries & secrets filtered)
├── assets/
│   ├── jarvis_hero_banner.png  # 3D Futuristic HUD Hero Banner
│   └── OpenJarvis_Circular_Logo.png
├── src/
│   └── openjarvis/
│       ├── agents/             # Native ReAct, CodeAct, and conversational agents
│       ├── tools/
│       │   └── deep_view.py    # Exclusive The Deep View retrieval engine & parser
│       ├── models/             # Ollama client with 16K context window enforcement
│       └── core/               # Context governors, tool registries, and execution loops
└── tests/
    └── test_deep_view.py       # Comprehensive verification suite for Deep View tools
```

---

## 🧪 Verification & Testing

Run the automated verification suite to validate tool execution, domain filters, and context management:

```bash
# Run Deep View retrieval & security tests
pytest tests/test_deep_view.py -v
```

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

---

<div align="center">
  <b>⚡ JARVIS &bull; Built with pride for autonomous, private, and factual AI.</b>
</div>
