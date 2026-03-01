<div align="center">

<img src="https://raw.githubusercontent.com/Saksham653/smartedge-copilot/main/assets/logo.png" width="120" alt="SmartEdge Copilot Logo" />

# ⚡ SmartEdge Copilot

### AI-Powered Research & Meeting Intelligence System

[![AMD Slingshot](https://img.shields.io/badge/AMD-Slingshot%202025-E8450A?style=for-the-badge&logo=amd&logoColor=white)](https://www.amd.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LLM%20Backend-F55036?style=for-the-badge)](https://groq.com)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**SmartEdge Copilot** transforms raw AI outputs into structured, persistent, and measurable intelligence — featuring research automation, meeting summarization, task extraction, unified knowledge search, and real-time analytics. All in a sleek dark dashboard.

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Usage](#-usage) • [API Docs](#-backend-api) • [Team](#-team)

</div>

---

## 🎯 What is SmartEdge Copilot?

SmartEdge Copilot is a full-stack AI productivity platform built for the **AMD Slingshot Hackathon 2025** by **Fixit Club**. It connects a powerful Python AI backend (powered by Groq's blazing-fast LLM inference) with a polished Streamlit dashboard — giving teams a single place to research, record meetings, manage tasks, and track AI performance.

> **Built for teams who want AI that actually remembers, organizes, and acts — not just answers.**

---

## ✨ Features

### 🏠 Command Center Dashboard
- Live KPI cards — total AI runs, tokens consumed, average latency, total cost
- Token usage over time chart
- Feature distribution breakdown (Research vs Meetings vs Optimizer)
- Real-time performance insights from the SQLite metrics database

### 🔍 Research Assistant
- Enter any topic → AI generates a **structured 4-section research report**
  - Concise Summary · Key Concepts · Practical Applications · References
- Prompt optimization layer automatically rewrites queries for token efficiency
- Every result **auto-saved** to SQLite with full metrics logging
- Search history with keyword filtering
- One-click **Markdown export** of any research note

### 🎙️ Meeting Intelligence
- Paste any meeting transcript → AI produces a **full structured summary**
  - Executive Summary · Key Topics · Action Items · Deadlines · Decisions
- **Automatic task extraction** — action items are parsed and saved directly to the tasks table
- Natural language deadline detection → converted to ISO `YYYY-MM-DD` format automatically
- Export any meeting note as a formatted Markdown file

### ✅ Task Management
- All tasks auto-populated from meeting analysis — zero manual entry
- Filter by status: `pending` / `done` / `cancelled`
- One-click status updates (✓ Done / ↩ Reopen)
- Visual completion progress bar
- Linked back to source meeting for full traceability

### 📚 Knowledge Hub
- **Unified semantic search** across all research notes AND meeting summaries
- Single query searches both tables simultaneously via `UNION ALL` SQL
- Preview snippets with source type indicators
- Export any result directly to Markdown

### 📊 Analytics
- Token usage bar chart over time
- Cost trend line chart
- Per-feature breakdown table: avg latency · total tokens · total cost · run count
- Performance insights: most expensive feature, slowest feature, cost per 1K tokens

---

## 🏗️ Architecture

```
smartedge-copilot/
│
├── smartedge_app.py          # Streamlit UI (6-page dashboard)
│
├── backend/
│   ├── database.py           # SQLite connection + schema init
│   ├── ai_wrapper.py         # Universal LLM caller (Groq / OpenAI / Ollama)
│   ├── optimizer.py          # Prompt optimization engine
│   ├── research.py           # Research generation pipeline
│   ├── research_db.py        # Research CRUD operations
│   ├── meeting.py            # Meeting summarization pipeline
│   ├── meeting_db.py         # Meeting CRUD + auto task creation
│   ├── tasks.py              # Task parser + DB operations
│   ├── knowledge_hub.py      # Unified cross-table search
│   ├── analytics_service.py  # Aggregated metrics & insights
│   ├── export_service.py     # Markdown export engine
│   ├── charts.py             # Matplotlib chart generation
│   ├── metrics.py            # Metrics helpers
│   ├── logger.py             # Logging utilities
│   └── utils.py              # Token counting, cost calc, helpers
│
├── data/
│   └── metrics.db            # SQLite database (auto-created on first run)
│
├── .env                      # Your API keys (not committed)
├── .env.example              # Template
├── requirements.txt
└── README.md
```

### Data Flow

```
User Input
    │
    ▼
┌─────────────────────────────────────────┐
│           Streamlit UI Layer            │
│         (smartedge_app.py)              │
└──────────────┬──────────────────────────┘
               │
    ┌──────────▼──────────┐
    │   Optimizer Layer   │  ← rewrites prompt for token efficiency
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │    AI Wrapper       │  ← calls Groq / OpenAI / Ollama
    │   (call_llm)        │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   Parser + DB       │  ← structured sections → SQLite
    │  research_db /      │
    │  meeting_db /       │
    │  tasks              │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  Analytics Engine   │  ← aggregates metrics table
    └─────────────────────┘
```

### Database Schema

```sql
-- Tracks all AI performance data
metrics         (id, feature_name, model, prompt_tokens, completion_tokens,
                 total_tokens, latency_ms, cost, created_at)

-- Stores structured research outputs
research_notes  (id, query, optimized_prompt, summary, key_concepts,
                 applications, references_text, total_tokens, latency_ms,
                 cost, model, created_at)

-- Stores structured meeting outputs
meeting_notes   (id, title, transcript, summary, key_topics, action_items,
                 deadlines, decisions, total_tokens, latency_ms, cost,
                 model, created_at)

-- Auto-extracted from meetings
tasks           (id, source_type, source_id, assignee, task_description,
                 deadline, status, created_at)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or newer
- A free [Groq API key](https://console.groq.com) (takes 1 minute)

### 1. Clone the repository

```bash
git clone https://github.com/Saksham653/smartedge-copilot.git
cd smartedge-copilot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your credentials:

```env
OPENAI_API_KEY=gsk_your_groq_key_here
OPENAI_BASE_URL=https://api.groq.com/openai/v1
MODEL=llama-3.1-8b-instant
PROVIDER=openai
```

> 💡 Get your free Groq API key at [console.groq.com](https://console.groq.com) → API Keys → Create API Key

### 4. Run the app

```bash
streamlit run smartedge_app.py
```

Your browser will open automatically at **http://localhost:8501** 🎉

---

## 🔑 Environment Variables

| Variable | Description | Example |
|---|---|---|
| `OPENAI_API_KEY` | Your Groq API key | `gsk_abc123...` |
| `OPENAI_BASE_URL` | LLM provider base URL | `https://api.groq.com/openai/v1` |
| `MODEL` | Model to use | `llama-3.1-8b-instant` |
| `PROVIDER` | Provider type | `openai` |

### Supported Models (Groq — all free)

| Model | Best For | Context |
|---|---|---|
| `llama-3.1-8b-instant` | Speed, everyday tasks | 128K |
| `llama-3.3-70b-versatile` | Complex reasoning | 128K |
| `mixtral-8x7b-32768` | Long meeting transcripts | 32K |
| `gemma2-9b-it` | Lightweight, fast | 8K |

---

## 📖 Usage

### Research Assistant

1. Navigate to **🔍 Research** in the sidebar
2. Type any topic in the search box
3. Click **GENERATE RESEARCH**
4. AI produces a structured 4-section report and saves it automatically
5. View history in the **History** tab — search by keyword, export to Markdown

### Meeting Summarizer

1. Navigate to **🎙️ Meetings**
2. Enter a meeting title
3. Paste your transcript (any format — bullet points, raw dialogue, notes)
4. Click **ANALYZE MEETING**
5. AI extracts: summary, topics, action items, deadlines, decisions
6. Tasks are **automatically created** in the Tasks page

### Task Management

1. Navigate to **✅ Tasks**
2. Tasks are auto-populated from meeting analysis
3. Use the filter dropdown to view by status
4. Click **✓ Done** to mark complete or **↩ Reopen** to revert

### Knowledge Hub

1. Navigate to **📚 Knowledge Hub**
2. Type a keyword to search across ALL research and meeting notes
3. Click **📥 Export** on any result to download as Markdown

---

## 🔌 Backend API

### Research

```python
from backend.research import generate_research

result = generate_research("AMD Instinct MI300X architecture")
# Returns:
# {
#   "summary": "...",
#   "key_concepts": "...",
#   "applications": "...",
#   "references": "...",
#   "metrics": { "total_tokens": 847, "latency_ms": 1240.5, "cost": 0.0017, "model": "..." }
# }
```

### Meeting Summarization

```python
from backend.meeting import generate_meeting_summary

result = generate_meeting_summary("Sprint Planning", transcript_text)
# Returns:
# {
#   "summary": "...",
#   "key_topics": "...",
#   "action_items": "- John — Fix API endpoint\n- Sara — Write tests",
#   "deadlines": "- Fix API endpoint — 2025-12-01",
#   "decisions": "...",
#   "metrics": { ... }
# }
```

### Knowledge Search

```python
from backend.knowledge_hub import search_knowledge_hub
from backend.database import DB_PATH

results = search_knowledge_hub(DB_PATH, "machine learning", limit=10)
# Returns list of: { "type": "research"|"meeting", "id": int, "title": str, "preview": str }
```

### Analytics

```python
from backend.analytics_service import get_overall_totals, generate_performance_insights

totals   = get_overall_totals()
insights = generate_performance_insights()
# totals:   { "total_runs": 42, "total_tokens": 85420, "avg_latency_ms": 1102.3, "total_cost": 0.17 }
# insights: { "most_expensive_feature": "research", "slowest_feature": "optimizer", ... }
```

### Export to Markdown

```python
from backend.export_service import export_note_markdown
from backend.database import DB_PATH

md = export_note_markdown(DB_PATH, "research", note_id=1)
# Returns formatted Markdown string ready to save
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **UI Framework** | Streamlit |
| **LLM Provider** | Groq (llama-3.1-8b-instant) |
| **AI Client** | OpenAI Python SDK (Groq-compatible) |
| **Database** | SQLite (via Python `sqlite3`) |
| **Charts** | Plotly |
| **Prompt Optimization** | Custom LLM-based optimizer |
| **Language** | Python 3.11+ |
| **Styling** | Custom CSS · Rajdhani + JetBrains Mono fonts |

---

## 📦 Dependencies

```
streamlit
plotly
pandas
openai
python-dotenv
tiktoken
requests
matplotlib
numpy
```

Install all at once:

```bash
pip install streamlit plotly pandas openai python-dotenv tiktoken requests matplotlib numpy
```

---

## 🧪 Running Tests

```bash
# Test search functionality
python test_search.py

# Test meeting automation (task extraction)
python test_meeting_automation.py

# Test export service
python test_export.py
```

---

## 🗺️ Roadmap

- [x] Research generation + persistent storage
- [x] Meeting summarization + auto task extraction
- [x] Unified knowledge hub search
- [x] Real-time analytics dashboard
- [x] Markdown export engine
- [x] Prompt optimization layer
- [ ] User authentication
- [ ] Multi-user workspace support
- [ ] Model-level comparison analytics
- [ ] REST API layer (FastAPI)
- [ ] Deployment (Docker + cloud)

---

## 👥 Team

**Fixit Club** — AMD Slingshot Hackathon 2025

| Role | Contribution |
|---|---|
| **Backend** | Python AI pipeline, SQLite architecture, LLM integration |
| **Frontend** | Streamlit dashboard, UI/UX, dark theme design |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ⚡ by Fixit Club for AMD Slingshot 2025**

*SmartEdge Copilot — Intelligence that persists.*

</div>
