# LLM-as-BT Planner

LLM-as-BT Planner turns natural-language task instructions into executable behavior trees.  
The update also adds an MRBTP-inspired multi-robot layer and a visual gridworld simulator for testing team behavior.

## What This Repository Does

It combines two ideas:

- **LLM-as-BT-Planner**: use an LLM to convert instructions into a small symbolic plan
- **MRBTP**: turn that plan into coordinated multi-robot behavior with explicit assignment, handoff, and reactive execution

The result is:

1. a user types an instruction
2. the LLM outputs a short symbolic action list
3. the compiler builds a reactive behavior tree
4. the multi-robot layer splits the work across robots
5. the gridworld simulator visualizes and tests the result

## Action Vocabulary

The planner uses a small fixed set of actions:

- `Pick`
- `MoveTo`
- `Place`
- `Insert`
- `ChangeTool`
- `Handoff`

This is deliberate: the LLM chooses **what** should happen, while deterministic code decides **how** to compile and execute it.

## How LLM-as-BT-Planner and MRBTP Are Joined

### In Plain Words

The two systems are not fused into one monolithic model.

- The **LLM-as-BT-Planner** part is the semantic front end. It reads the instruction and produces a compact symbolic plan.
- The **MRBTP-inspired** part is the execution backbone. It takes that symbolic plan, splits it into goal-carrying chunks, assigns them to robots, and builds a coordinated team behavior tree.

So the LLM does not directly write a full multi-robot tree.  
Instead:

- the LLM says: `Pick`, `MoveTo`, `Place`, `Handoff`, `Insert`, ...
- the MRBTP layer decides which robot should do which chunk
- the BT compiler turns those chunks into reactive trees with conditions and recovery logic

### Technically

The main join points are:

```python
from src.llm_client import LLMTaskPlanner
from src.bt_builder import build_tree_from_json
from src.multi_robot_planner import build_multi_robot_tree_from_json, resolve_robot_profiles

planner = LLMTaskPlanner()
plan = planner.plan_task("Pick the gear and insert it into the chassis.")

# Single-robot reactive BT
tree = build_tree_from_json(plan)

# Multi-robot reactive BT
profiles = resolve_robot_profiles()
team_tree = build_multi_robot_tree_from_json(plan, robot_profiles=profiles)
```

For the testing simulator, the join is similar but scenario-oriented:

```python
from src.gridworld_env import build_env_from_typed_scenario
from src.llm_client import LLMTaskPlanner

planner = LLMTaskPlanner()
env = build_env_from_typed_scenario(
    scenario_text="All robots collect circles and place them in different corners.",
    num_robots=3,
    num_circles=3,
    layout_name="open_room",
    planner=planner,
)
```

In the multi-robot path, the repo adds:

- plan segmentation into goal chunks
- phase grouping for parallel execution
- robot capability filtering
- explicit `Handoff` support
- tool-aware `Insert` support
- intention-aware backup suppression

## Main Features

- natural language -> symbolic plan -> reactive BT
- single-robot and multi-robot execution paths
- human-in-the-loop revision (`scheme3`)
- recursive planning (`scheme4`)
- explicit handoff and tool-change support
- LLM-driven visual gridworld simulator
- transport and assembly-style gridworld tasks with gears, shafts, fixtures, and tool-aware inserts
- offline demo scenarios
- unit tests for planner, compiler, multi-robot behavior, and gridworld execution

## Quick Start

If you only want to clone the repo and run the visual simulator:

### 1. Clone

```powershell
git clone https://github.com/YevgeniyDev/llm-as-bt-planner.git
cd llm-as-bt-planner
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Create `.env`

```powershell
Copy-Item .env.example .env
```

Then put real credentials in `.env`.

Minimal Hugging Face setup:

```env
LLM_PROVIDER=huggingface
HF_TOKEN=your_real_huggingface_token
HUGGINGFACE_MODEL=Qwen/Qwen2.5-7B-Instruct
```

Minimal OpenAI setup:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_real_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

### 5. Launch the visual simulator

```powershell
python -m src.demo_scenarios typed_gridworld
```

That opens the Tkinter testing window.

## Simulator Usage

The visual simulator is the easiest way to test the project.

You can:

- choose a preset or `Custom`
- type your own instruction
- choose robot count
- choose object count
- choose a layout
- run and replay the scenario
- inspect actions, object state, plan, and BT

The simulator supports both:

- transport-style objects such as circles, squares, and triangles
- assembly-style objects such as gears, shafts, pins, and plates, plus fixed locations such as `chassis`, `bearing_block`, `panel_slot`, `tool_rack`, and `precision_station`

### Default behavior

- If you select `Custom` and leave the placeholder unchanged, the simulator runs the default scenario.
- The `Summary` panel is hidden by default.

### Presets

- `Custom` - write your own instruction; if you leave the placeholder unchanged, the default scenario runs
- `Distributed Corners` - each robot takes a circle and delivers it to a different corner
- `Two Move One Waits` - two robots work on delivery while a third stays idle at the center
- `Stationary Receiver` - two robots bring circles one by one to a waiting receiver robot
- `Three Robot Relay` - one circle is passed across three robots before its final placement
- `Parallel Rooms` - robots distribute circles across separate rooms with parallel work
- `Wall Sweep` - eight robots place eight circles along the room walls in a larger map
- `Precision Insert` - one robot changes to a precision tool and inserts a gear into the chassis
- `Relay Insert` - one robot hands off a shaft and another robot changes tools before inserting it
- `Three Robot Assembly` - two feeder robots fetch parts in parallel while one assembler prepares and inserts both parts
- `Four Robot Assembly` - two feeder robots and two assembler robots cooperate on a dual-part assembly with parallel final inserts

### Layouts

- `open_room` - one open area with no internal walls; best for simple coordination tests
- `split_room` - two main areas connected by doorways; useful for routing and partial separation
- `four_rooms` - a four-room map with connecting openings; good for multi-region delivery and relay tasks
- `handoff_hall` - rooms linked by a central corridor; designed for handoff and receiver-style scenarios

## Common Commands

### Open the visual simulator

```powershell
python -m src.demo_scenarios typed_gridworld
```

### Open the simulator with a predefined scenario

```powershell
python -m src.demo_scenarios typed_gridworld --scenario-text "All robots collect circles and place them along the walls." --num-robots 8 --num-circles 8 --layout four_rooms
```

### Run the simulator in text-only mode (doesn't open simulator window, instead shows it in terminal)

```powershell
python -m src.demo_scenarios typed_gridworld --text-only
```

### Run the main planner pipeline

```powershell
python -m src.main
```

Use this when you want to test the normal text-to-BT flow outside the gridworld simulator. The command prompts for an instruction in the terminal, falls back to the default example if you leave it empty, sends the task to the configured LLM, validates the generated symbolic plan, builds the behavior tree, and executes it with the project runtime. This path requires a valid provider setup in `.env`.

### Run the full test suite

```powershell
python -m unittest discover -s tests -v
```

Use this after setup or after making code changes. It runs the automated test modules for the planner, LLM client, gridworld environment, GUI helpers, validation logic, and reactive BT behavior, then prints a verbose pass/fail result for each test. This is the fastest way to confirm the repo is installed correctly and that recent changes did not break existing behavior.

### Run offline demos

These do **not** require an LLM call.

```powershell
python -m src.demo_scenarios dynamic_failure
python -m src.demo_scenarios multi_robot_parallel
python -m src.demo_scenarios heterogeneous_handoff
```

These are small deterministic examples for debugging the symbolic BT layer without depending on external APIs or the visual simulator:

- `dynamic_failure` simulates a single-robot recovery case where an object is dropped mid-task and the reactive BT must recover.
- `multi_robot_parallel` shows two robots executing different goal segments in parallel under the multi-robot planner.
- `heterogeneous_handoff` shows capability-aware cooperation, where one robot hands off an object and another changes tools before finishing the task.

## Configuration

Important `.env` variables:

| Variable                   | Purpose                                            |
| -------------------------- | -------------------------------------------------- |
| `LLM_PROVIDER`             | `huggingface` or `openai`                          |
| `HF_TOKEN`                 | Hugging Face token                                 |
| `HUGGINGFACE_MODEL`        | Hugging Face model id                              |
| `OPENAI_API_KEY`           | OpenAI key                                         |
| `OPENAI_MODEL`             | OpenAI model id                                    |
| `PLANNING_SCHEME`          | `scheme3` or `scheme4`                             |
| `ENABLE_HUMAN_IN_THE_LOOP` | enables manual review for `scheme3`                |
| `MAX_REVIEW_ROUNDS`        | max revision rounds for `scheme3`                  |
| `MAX_RECURSION_DEPTH`      | recursion depth for `scheme4`                      |
| `MAX_SUBGOALS_PER_LEVEL`   | recursive branching cap                            |
| `ENABLE_MULTI_ROBOT`       | enables the multi-robot path in `src.main`         |
| `MULTI_ROBOT_ROBOTS`       | optional JSON robot-team config for `src.main`     |
| `GRIDWORLD_REPAIR_ROUNDS`  | LLM repair retries for invalid simulator scenarios |

Notes:

- The visual simulator uses its own robot specs produced by the LLM scenario planner.
- `ENABLE_MULTI_ROBOT` matters for `src.main`, not for the gridworld tester.

## What To Test First

Recommended order:

1. `python -m src.demo_scenarios typed_gridworld`
2. `python -m unittest discover -s tests -v`
3. `python -m src.demo_scenarios multi_robot_parallel`
4. `python -m src.main`

This checks:

- the GUI opens
- your API credentials work
- the simulator runs
- the unit tests pass
- the main instruction-to-BT pipeline works

## Troubleshooting

### The visual window does not open

- On Windows, Tkinter is usually included with standard Python.
- On Linux, you may need `python3-tk`.
- If you are on a headless machine, use:

```powershell
python -m src.demo_scenarios typed_gridworld --text-only
```

### The simulator falls back to text mode immediately

- Check `.env`
- Make sure your token is real, not a placeholder
- Re-run after updating credentials

### The LLM produces an invalid gridworld scenario

The simulator already tries repair rounds via `GRIDWORLD_REPAIR_ROUNDS`, but badly underspecified instructions can still fail.  
When that happens, make the instruction more explicit about:

- which robots move
- who waits
- where handoffs happen
- final target locations

## Key Files

- `src/llm_client.py` - LLM planning and repair prompts
- `src/bt_builder.py` - single-robot BT compiler
- `src/multi_robot_planner.py` - MRBTP-style team BT compiler
- `src/gridworld_env.py` - gridworld execution environment
- `src/gridworld_app.py` - visual simulator UI
- `src/gridworld_presets.py` - simulator presets
- `src/demo_scenarios.py` - demos and simulator launcher

## Limits

- The action vocabulary is intentionally small.
- The world is symbolic, not a real robot middleware stack.
- The multi-robot layer is strongest on transport, handoff, and insertion-style tasks.
- Heterogeneous collaboration depends on explicit symbolic actions such as `Handoff` and `ChangeTool`.
