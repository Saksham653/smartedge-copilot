SmartEdge Copilot — Backend Architecture

AI-powered research and meeting intelligence system with persistent storage, task automation, and performance analytics.

This repository currently contains the complete backend implementation.

1. System Overview

SmartEdge Copilot transforms AI outputs into structured, persistent, and measurable intelligence.

The backend provides:

Structured research storage

Structured meeting storage

Automatic task extraction

Unified knowledge search

Markdown export

Performance metrics logging

Aggregated analytics and insights

All data is stored in SQLite.

2. Architecture
backend/
│
├── database.py
├── research_db.py
├── meeting_db.py
├── tasks.py
├── knowledge_hub.py
├── export_service.py
├── analytics_service.py

The system follows a modular architecture with clear separation of concerns.

3. Database Layer
database.py

Responsibilities:

Resolve project root

Manage SQLite connection

Initialize database schema

Tables
metrics

Tracks AI performance usage.

Fields:

id

feature_name

model

prompt_tokens

completion_tokens

total_tokens

latency_ms

cost

created_at

research_notes

Stores structured research outputs.

meeting_notes

Stores structured meeting outputs.

tasks

Stores tasks extracted from meetings.

4. Research Module
research_db.py
save_research_note()

Inserts structured research into research_notes

Logs performance into metrics

Data persisted:

query

optimized_prompt

summary

key_concepts

applications

references_text

total_tokens

latency_ms

cost

model

5. Meeting Module
meeting_db.py
save_meeting_note()

Inserts structured meeting data into meeting_notes

Logs metrics into metrics

Automatically extracts tasks from action_items

Stores tasks in tasks table

Execution flow:

Insert meeting_note
→ Insert metrics
→ Extract tasks
→ Insert tasks
→ Commit
6. Task Automation Engine
tasks.py
extract_tasks_from_action_items()

Parses:

Bulleted lines

Numbered lists

Optional assignee separator

Optional deadline using "by"

Returns structured tasks:

assignee

task_description

deadline

create_tasks_from_meeting()

Inserts parsed tasks into tasks

Links via:

source_type = "meeting"

source_id = meeting_id

7. Knowledge Hub
knowledge_hub.py
search_knowledge_hub()

Performs unified search across:

research_notes

meeting_notes

Uses raw SQL with UNION ALL and LIKE matching.

Returns normalized results:

type

id

title

preview

8. Export Engine
export_service.py
export_note_markdown()

Retrieves research or meeting note

Formats structured Markdown

Returns Markdown string

Raises custom error if note not found

9. Analytics Engine
analytics_service.py
Aggregation Functions

get_feature_summary()

get_overall_totals()

get_usage_over_time()

Uses SQL aggregation:

AVG()

SUM()

COUNT()

DATE grouping

Insights Function

generate_performance_insights()

Computes:

Most expensive feature

Slowest feature

Cost per 1K tokens

Average cost per run

Handles division-by-zero safely.

10. End-to-End Data Flow

Research Flow:

AI Output
→ save_research_note()
→ research_notes insert
→ metrics insert

Meeting Flow:

AI Output
→ save_meeting_note()
→ meeting_notes insert
→ metrics insert
→ task extraction
→ tasks insert

Analytics Flow:

metrics table
→ SQL aggregation
→ structured summary
→ performance insights
11. Design Decisions

SQLite chosen for simplicity and portability

Raw SQL used (no ORM) for transparency

Modular separation for extensibility

Metrics decoupled from feature tables

Schema supports future model comparison

12. Current Status

Backend complete.

Implemented:

Persistent storage

Task automation

Unified search

Markdown export

Performance logging

Insight generation

Pending:

UI integration

Dashboard visualization

Deployment

Model-level comparison analytics
