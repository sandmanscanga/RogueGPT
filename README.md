# 🧠 RogueGPT - gptcli — Command-Line ChatGPT for Linux/WSL

`gptcli` is a compact but feature-rich **ChatGPT terminal client** written in Python.
It offers **per-directory threads**, **colored streaming output**, **search/export utilities**, and **persistent conversation history** — all from your shell.

---

## 📦 Features

* Centralized history under `~/.config/gptcli/threads`
* Automatically keeps a separate thread for each working directory
* Streaming output with rich color themes
* Thread management: list, show, retitle, start new, or continue existing
* Full conversation persistence (like the ChatGPT web app)
* Smart **brevity mode** for `--max-tokens`
* Search all past threads (supports regex, JSON output, and context windows)
* Export threads to Markdown, plain text, or JSON
* Filter by message role or content pattern
* Works out of the box with WSL and Debian

---

## ⚙️ Installation

### 1. Install dependencies

```bash
python3 -m pip install --upgrade openai rich
```

### 2. Add your API key

```bash
echo 'export OPENAI_API_KEY=YOUR_KEY_HERE' >> ~/.zshrc
source ~/.zshrc
```

### 3. Install the script

```bash
mkdir -p ~/.local/bin
nano ~/.local/bin/gpt
chmod +x ~/.local/bin/gpt
```

Paste the contents of `gptcli.py` and save.

### 4. Optional environment variables

| Variable         | Description                    | Default            |
| ---------------- | ------------------------------ | ------------------ |
| `OPENAI_API_KEY` | Your OpenAI API key (required) | —                  |
| `GPT_MODEL`      | Default model to use           | `gpt-5`            |
| `GPT_HOME`       | Config directory for threads   | `~/.config/gptcli` |

---

## 🧭 Basic Usage

### Ask a question

```bash
gpt "What is a symbolic link?"
```

### Continue the same thread (in this directory)

```bash
gpt -r "Give me examples."
```

### Start a new thread

```bash
gpt -n "Help me debug a Flask app."
```

### From stdin

```bash
echo "Summarize this text..." | gpt
```

### Set system behavior

```bash
gpt -s "You are a Linux expert." "Explain chmod permissions."
```

---

## 💬 Conversation Management

### List all threads

```bash
gpt threads
```

### Show messages in a thread

```bash
gpt show --thread <THREAD_ID> -n 20
```

### Retitle a thread

```bash
gpt retitle --thread <THREAD_ID> "Docker Build Optimization"
```

Threads are stored under:

```
~/.config/gptcli/threads/<THREAD_ID>.jsonl
```

---

## 🔍 Search Your Archive

### Basic substring search

```bash
gpt search -q "docker"
```

### Regex search

```bash
gpt search -q "^(ffmpeg|yt)" -r
```

### Search only assistant replies

```bash
gpt search -q "apt lock" --role assistant
```

### Limit and context

```bash
gpt search -q "ssh" -l 10 -c 2
```

### JSON output (for scripting)

```bash
gpt search -q "systemd" --json | jq '.match.content'
```

---

## 📤 Export Conversations

You can export entire threads in different formats.

### Markdown

```bash
gpt export --thread <THREAD_ID> --to md --out mychat.md
```

### Plain text

```bash
gpt export --thread <THREAD_ID> --to txt
```

### JSON (raw structure)

```bash
gpt export --thread <THREAD_ID> --to json > thread.json
```

### Filter by role

```bash
gpt export --thread <THREAD_ID> --role assistant
```

### Filter by content (substring)

```bash
gpt export --thread <THREAD_ID> --filter "docker"
```

### Filter by regex

```bash
gpt export --thread <THREAD_ID> --filter "^error:" --regex
```

---

## 🧩 Advanced Options

### Token control

Use `--max-tokens` to limit the model’s output size.

```bash
gpt "Explain tmux vs screen" --max-tokens 200
```

If the limit is reached:

> ⚠️ “Reply hit --max-tokens limit and was truncated.”

To get a naturally concise response instead of a hard cutoff, `gptcli` automatically adds a **conciseness hint** when you use this flag.

---

## 🧱 Architecture Overview

* **`registry.json`**
  Maps directories to thread IDs for per-project continuity.

* **Thread files** (`.jsonl`)
  Store each message (user + assistant) sequentially.

* **Metadata files** (`.meta.json`)
  Store title, creation date, model, and other metadata.

* **Color-coded roles**

  * 🟡 User
  * 🟢 Assistant
  * 🟣 System
  * ⚪ Metadata and hints

---

## 🧠 Examples

### 1. Quick Linux help

```bash
gpt "How do I kill a process by port number?"
```

### 2. Continue conversation

```bash
gpt -r "Can I make that permanent?"
```

### 3. Search and export results

```bash
gpt search -q "kill process" --json | jq '.title'
gpt export --thread <THREAD_ID> --to md --role assistant > summary.md
```

### 4. Copy output to clipboard (WSL)

```bash
gpt "convert mp4 to gif with ffmpeg" | clip.exe
```

---

## 🧩 Future Ideas

Potential enhancements:

* `--since/--until` filters for date-based search/export
* `--json` option for chat replies
* `gpteval`: run model-suggested shell commands safely
* `gpte`: edit last reply interactively before sending

---

## 🏁 License

MIT License © 2025
You are free to modify and redistribute. Attribution appreciated.
