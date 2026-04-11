# LLM-as-BT Planner Prototype

## Overview

This repository is a compact Python prototype inspired by the paper *LLM-as-BT-Planner: Leveraging LLMs for Behavior Tree Generation in Robot Task Planning*. The goal is to demonstrate the paper's core systems idea in a clean, inspectable form:

1. A user provides a natural-language task instruction.
2. An LLM converts that instruction into a structured symbolic plan.
3. The symbolic plan is translated into an executable Behavior Tree (BT).
4. Mock robot actions execute the BT to show end-to-end task flow.

Rather than reproducing a full robotics stack, this project focuses on the planning architecture itself. That makes it well-suited for a portfolio or research discussion because the mapping from language to symbolic control is visible in only a few modules.

## Why This Design

- `llm_client.py` isolates prompt engineering and API interaction so the planning interface can evolve independently of BT execution.
- `bt_builder.py` keeps the symbolic-to-executable translation explicit, which is useful when discussing interpretability and modularity in BT-based planning.
- `robot_actions.py` uses mock `py_trees` behaviors instead of real hardware drivers so the prototype remains runnable on a laptop.
- The root BT node is a `Sequence(memory=True)` because task plans are ordered procedures. Once a child succeeds, the executor should continue from the next step instead of re-ticking previously completed actions.

## Repository Layout

```text
llm_bt_planner/
├── src/
│   ├── __init__.py
│   ├── bt_builder.py
│   ├── llm_client.py
│   ├── main.py
│   └── robot_actions.py
├── .env.example
├── requirements.txt
└── README.md
```

## Planning Format

The LLM produces a JSON plan object with a `plan` array. Each entry is an action step such as:

```json
{
  "plan": [
    {"action": "Pick", "object": "gear"},
    {"action": "MoveTo", "target": "chassis"},
    {"action": "Insert", "object": "gear", "target": "chassis"}
  ]
}
```

Inside the code, the wrapper object is immediately unwrapped so the BT builder operates on the ordered action list. The wrapper also keeps the planner output consistent across both OpenAI and Hugging Face model paths.

## Setup Instructions

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env`, then add your Hugging Face token:

```env
LLM_PROVIDER=huggingface
HF_TOKEN=your_huggingface_token_here
HUGGINGFACE_MODEL=Qwen/Qwen2.5-7B-Instruct
```

The default Hugging Face model is [`Qwen/Qwen2.5-7B-Instruct`](https://hf.co/Qwen/Qwen2.5-7B-Instruct), which is a strong open instruct model and works through the Hugging Face router. This project uses the OpenAI Python SDK against Hugging Face's OpenAI-compatible endpoint at `https://router.huggingface.co/v1`, following Hugging Face's current integration guidance.

If you prefer OpenAI, set:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

## How to Run

Run the default demonstration task:

```powershell
python -m src.main
```

This uses the example instruction:

```text
Pick up the gear, move to the chassis, and insert the gear.
```

You can also pass a custom instruction:

```powershell
python -m src.main "Pick the bolt and place it on the tray."
```

## Expected Execution Flow

When you run the demo, the program will:

1. Load the provider, API key, and model name from `.env`
2. Send the task instruction to the LLM with a few-shot planning prompt
3. Print the generated JSON plan
4. Build a `py_trees` sequence from that plan
5. Tick the tree until the root returns `SUCCESS` or `FAILURE`

Because the mock robot actions take two ticks, the console output makes BT progression easy to follow during a demo.

## Module Notes

### `src/llm_client.py`

This module uses in-context examples to keep the LLM aligned with the exact action vocabulary understood by the BT builder. It now supports both OpenAI and Hugging Face-backed chat models while preserving the same downstream plan contract.

### `src/bt_builder.py`

This module performs the critical translation step from JSON to executable `py_trees` nodes. It also validates action names and required parameters so malformed plans fail early and transparently.

### `src/robot_actions.py`

These classes are intentionally lightweight. Each action reports `RUNNING` on its first tick and `SUCCESS` on its second tick, which better illustrates how behavior trees progress over time than instant success would.

### `src/main.py`

This is the integration entry point. It keeps the runtime loop simple on purpose so the planning-to-execution pipeline stays easy to explain in an academic or interview setting.

## Possible Extensions

- Add condition nodes for preconditions such as object availability or gripper state
- Replace mock actions with ROS service calls or motion-planning interfaces
- Expand the LLM schema to include action arguments like force thresholds or grasp strategies
- Add retry and fallback subtrees for more robust failure handling
- Evaluate plan quality against a benchmark task set inspired by the paper

## Notes

This repository is best understood as a software prototype for demonstrating research understanding, not a full reproduction of the paper's experimental setup. Its value is in making the language-to-BT pipeline concrete, modular, and easy to discuss with a prospective advisor or collaborator.
