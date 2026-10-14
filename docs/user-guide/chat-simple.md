# Simple Chat

A lightweight conversational AI with no tools and no agent overhead. This is the simplest possible OpenJarvis setup: just Ollama and a local model. Ideal for general-purpose chat, Q&A, brainstorming, and getting started quickly.

## Quickstart (3 minutes)

### 1. Install Ollama and pull a model

```bash
# Install Ollama: https://ollama.com
ollama pull qwen3.5:4b
```

### 2. Install and initialize OpenJarvis

```bash
git clone https://github.com/open-jarvis/OpenJarvis.git
cd OpenJarvis
uv sync
jarvis init --preset chat-simple --force
```

### 3. Ask a question

```bash
jarvis ask "What is quantum computing?"
```

That's it. No API keys, no tools, no cloud -- just a local model answering your questions.

## CLI Commands

```bash
# Single question
jarvis ask "Explain the difference between TCP and UDP"

# Interactive chat session (multi-turn conversation)
jarvis chat

# Start the API server for the browser or desktop app
jarvis serve

# Override the model for a single query
jarvis ask -m qwen3.5:9b "Explain general relativity"

# Adjust temperature (0.0 = deterministic, 1.0 = creative)
jarvis ask -t 0.2 "List the planets in our solar system"

# Output raw JSON
jarvis ask --json "What is 2+2?"
```

## Configuration Reference

The preset writes this to `~/.openjarvis/config.toml`:

```toml
[engine]
default = "ollama"

[intelligence]
default_model = "qwen3.5:4b"       # Fast and lightweight
# default_model = "qwen3.5:9b"     # Better quality
# default_model = "llama3.1:8b"    # Alternative model

[agent]
default_agent = "simple"            # Single-turn, no tools

[server]
host = "0.0.0.0"
port = 8000
```

### Model options

| Model | Parameters | Speed | Quality | Best for |
|-------|-----------|-------|---------|----------|
| `qwen3.5:4b` | 4B | Fast | Good | Quick answers, lightweight hardware |
| `qwen3.5:9b` | 9B | Balanced | Better | General-purpose chat, explanations |
| `qwen3.5:35b` | 35B | Slower | Best | Complex reasoning, detailed analysis |
| `llama3.1:8b` | 8B | Balanced | Good | Alternative if you prefer Meta models |

To switch models, either edit `~/.openjarvis/config.toml` or override per-query:

```bash
jarvis ask -m qwen3.5:35b "Write a detailed comparison of REST and GraphQL"
```

To pull a new model:

```bash
ollama pull qwen3.5:35b
```

## Using the Browser App

Start the backend server and the React frontend with one command:

```bash
./scripts/quickstart.sh
```

This opens [http://localhost:5173](http://localhost:5173) in your browser with a full chat interface, streaming responses, and an energy monitoring dashboard.

To run just the API server (for use with the desktop app or external clients):

```bash
jarvis serve
```

The server is OpenAI-compatible, so any client that works with the OpenAI API can point to `http://localhost:8000/v1`.

## Connecting Colibri

OpenJarvis can use a Colibri instance that exposes OpenAI-compatible Chat
Completions and model-list endpoints. In desktop **Settings → Inference source**,
choose **Custom OpenAI-compatible server**, then click **Use Colibri preset**.
Configure Colibri separately to bind to loopback on port `8001` (Jarvis uses port
`8000` by default). Enter the exact model ID reported by Colibri, add its API key
only if Colibri requires one, save, and restart OpenJarvis. The preset only sets
OpenJarvis's connection target; it does not install or start Colibri. To find the
model ID, request `http://127.0.0.1:8001/v1/models` (for example,
`curl http://127.0.0.1:8001/v1/models`) and copy the exact `id` value. Enter the
base URL as either `http://127.0.0.1:8001` or `http://127.0.0.1:8001/v1`; the
desktop normalizes either form so requests use one `/v1` prefix. Optional API
keys are stored in desktop secure storage and sent as bearer authorization.
Keep Colibri bound to loopback for local use; the preset does not change its bind
address or expose it to the network.

This connects Colibri for inference; OpenJarvis remains responsible for running
its own configured tools when the selected agent requests them. Colibri's optional
MCP server manages Colibri's lifecycle and is not itself a tool executor. Colibri
serves one active model at a time, so multi-model routing remains an OpenJarvis
capability and is not provided by this connection.

Ollama is already a supported local inference source and its installed models are
listed in Settings. Maya AI is not integrated: the reviewed assistant script does
not expose a supported service or MCP tool interface and contains direct OS
actions. Do not run or wrap that script as an OpenJarvis tool.

## Voice, camera, and offline use

Microphone transcription requires the Speech-to-Text setting and a configured
speech backend. Offline transcription requires a local backend and model; a remote
provider can send audio off-device. Camera hand tracking is optional in
**Settings → Hand tracking → Open hand tracking in chat**, or the **Hand tracking**
button above the chat input. Opening the panel does not start capture. It detects
landmarks and basic gestures locally, and stops when stopped, when the panel closes,
when you leave chat, or when the app/tab is hidden. Returning does not restart the
camera. The first camera action briefly requests permission to reveal
camera names, then immediately stops that temporary stream. Select the intended
device (for example, **Integrated Webcam**) from the camera list before pressing
**Start camera**. Open palm can optionally focus the chat input; gestures never
submit messages or call tools. It uses the bundled MediaPipe Tasks Vision runtime
and [Hand Landmarker model](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker).
The browser's online indicator is only a connectivity hint, not proof that a
particular server is reachable.

Local inference and cached interface assets do not make the assistant universally
autonomous. No financial or business-venture integration is configured here, and
OpenJarvis cannot guarantee legal compliance or revenue. It does not autonomously
retrain model weights; its existing memory, traces, and learning policies are
separate capabilities. MCP tools should be narrowly allowlisted, with integration-
level approval and audit controls for financial transactions, contracts, public
communications, and other irreversible actions.

## Using the Desktop App

1. Start the backend: `jarvis serve` (or `./scripts/quickstart.sh`)
2. Download and open the desktop app from the [releases page](https://github.com/open-jarvis/OpenJarvis/releases)
3. The app connects to `http://localhost:8000` automatically

## Switching Models

You can change the default model at any time:

**Edit the config:**

```bash
# Open the config file
${EDITOR:-nano} ~/.openjarvis/config.toml
# Change default_model to your preferred model
```

**Pull and switch in one step:**

```bash
ollama pull deepseek-r1:14b
jarvis ask -m deepseek-r1:14b "Hello"
```

**Use an environment variable:**

```bash
OPENJARVIS_MODEL=qwen3.5:9b jarvis ask "Hello"
```

## Troubleshooting

**"No running engine found"** -- Make sure Ollama is running. Start it with `ollama serve` or open the Ollama desktop app.

**"Model not found"** -- Pull the model first with `ollama pull <model-name>`. List available models with `ollama list`.

**Slow responses** -- Use a smaller model (`qwen3.5:4b`). Check available memory; models need RAM roughly equal to their parameter count in GB (e.g., 9B model needs ~9 GB).

**Want to add tools later?** -- Switch to the [Code Assistant](code-assistant.md) or [Deep Research](deep-research.md) config. Simple chat is intentionally minimal.

**Browser app not loading** -- Make sure both the backend (`jarvis serve`) and frontend are running. The `./scripts/quickstart.sh` script starts both automatically.
