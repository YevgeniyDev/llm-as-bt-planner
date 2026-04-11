# LLM-as-BT Planner

This repository is a compact prototype inspired by the paper _LLM-as-BT-Planner: Leveraging LLMs for Behavior Tree Generation in Robot Task Planning_.

The main idea is simple: take a natural-language instruction, turn it into a symbolic plan with an LLM, compile that plan into a reactive Behavior Tree, and execute it in a small mock robot world. The point of the project is not to be a full robotics stack. The point is to make the planning logic visible, testable, and easy to discuss.

Right now the project includes:

- reactive BT compilation with condition and fallback nodes
- Scheme 3 from the paper - "human-in-the-loop revision"
- Scheme 4 from the paper - "recursive planning with predicted-state rollouts"
- pre-execution validation for contradictory plans
- reproducible tests and demos

## What the Project Actually Does

The LLM does not generate raw tree control flow directly. It generates a simple symbolic action plan using a small fixed vocabulary:

- `Pick`
- `MoveTo`
- `Place`
- `Insert`

That symbolic plan is then compiled into a reactive `py_trees` Behavior Tree.

For example, a symbolic step like:

```text
Pick(gear)
```

becomes a reactive subtree like:

```text
Fallback(
  Holding(gear),
  Pick(gear)
)
```

That separation matters. It keeps the LLM output simple, while the control logic stays deterministic and inspectable.

## Why the Reactive BT Matters

This is not just a script that runs actions in order.

The root node is a memoryless `Sequence`, so the tree checks earlier conditions again on every tick. That gives the system three useful behaviors:

1. If a precondition is already satisfied, the action is skipped.
2. If the world changes during execution, the tree can recover.
3. If the symbolic plan is contradictory, the system exposes the contradiction instead of silently hiding it.

For `Place(gear, table)`, the compiler produces logic like this:

```text
Fallback(
  ObjectAt(gear, table),
  Sequence(
    Fallback(Holding(gear), Pick(gear)),
    Fallback(At(table), MoveTo(table)),
    Place(gear, table)
  )
)
```

That is the core difference between a procedural demo and a reactive planning prototype.

## Implemented Features

### Scheme 3: Human-in-the-loop

The system can:

1. ask the LLM for a symbolic plan
2. render the reactive BT preview
3. pause for human feedback
4. ask the LLM to revise the plan
5. execute the accepted result

This is useful when you want to inspect the tree before runtime or correct the plan without editing code.

### Scheme 4: Recursive planning

The recursive mode follows the same high-level logic as the paper:

- `MakePlan`: decide whether the task is primitive or should be decomposed
- `MakeTree`: recursively expand subgoals
- `PredictState`: roll the symbolic world state forward after each subproblem

For multi-step tasks, this produces a recursive trace before execution.

### Pre-execution validation

The project also checks for a specific class of reactive plan mistakes before execution starts.

Example:

```json
[
  { "action": "Pick", "object": "gear" },
  { "action": "MoveTo", "target": "chassis" },
  { "action": "Insert", "object": "gear", "target": "chassis2" }
]
```

This looks small, but in a reactive memoryless BT it can cause oscillation:

- one part of the tree keeps enforcing `At(chassis)`
- another part keeps enforcing `At(chassis2)`

The validator warns about that before execution.

## Project Structure

```text
llm-as-bt-planner/
|-- src/
|   |-- __init__.py
|   |-- bt_builder.py
|   |-- demo_scenarios.py
|   |-- llm_client.py
|   |-- main.py
|   |-- plan_validator.py
|   |-- recursive_planner.py
|   `-- robot_actions.py
|-- tests/
|   |-- test_llm_client.py
|   |-- test_plan_validator.py
|   `-- test_reactive_bt.py
|-- .env.example
|-- requirements.txt
`-- README.md
```

## Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Create `.env`

Copy `.env.example` to `.env`, then add your real token.

Important detail:

- `.env.example` is only a template
- the program actually reads `.env`
- shell environment variables also count
- `load_dotenv()` does not override values already set in your shell

Minimal Hugging Face setup:

```env
LLM_PROVIDER=huggingface
HF_TOKEN=your_real_huggingface_token
HUGGINGFACE_MODEL=Qwen/Qwen2.5-7B-Instruct
PLANNING_SCHEME=scheme3
ENABLE_HUMAN_IN_THE_LOOP=true
```

Optional OpenAI setup:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_real_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

## Configuration

| Variable                   | Meaning                                  |
| -------------------------- | ---------------------------------------- |
| `LLM_PROVIDER`             | `huggingface` or `openai`                |
| `HF_TOKEN`                 | Hugging Face token for router access     |
| `HUGGINGFACE_MODEL`        | model id used for Hugging Face inference |
| `OPENAI_API_KEY`           | OpenAI API key                           |
| `OPENAI_MODEL`             | OpenAI model id                          |
| `PLANNING_SCHEME`          | `scheme3` or `scheme4`                   |
| `ENABLE_HUMAN_IN_THE_LOOP` | enables the interactive review step      |
| `MAX_REVIEW_ROUNDS`        | max revision rounds in Scheme 3          |
| `MAX_RECURSION_DEPTH`      | recursion depth for Scheme 4             |
| `MAX_SUBGOALS_PER_LEVEL`   | subgoal cap per recursive layer          |

## How to Run

### Default run

```powershell
python -m src.main
```

### Run Scheme 3 explicitly

```powershell
$env:PLANNING_SCHEME='scheme3'
$env:ENABLE_HUMAN_IN_THE_LOOP='true'
python -m src.main "Assemble the gearbox."
```

If you reject the first plan, keep the feedback inside the current action vocabulary. A good correction is:

```text
Replace chassis with box everywhere.
```

A request like:

```text
Change the tool first.
```

is outside the current symbolic skill set, because the prototype does not implement a `ChangeTool` primitive.

### Run Scheme 4 explicitly

```powershell
$env:PLANNING_SCHEME='scheme4'
$env:ENABLE_HUMAN_IN_THE_LOOP='false'
python -m src.main "Pick up the screwdriver. Move to the panel. Place the screwdriver on the panel. Pick up the hammer. Move to the table. Place the hammer on the table. Pick up the gear. Move to the chassis. Insert the gear into the chassis."
```

You should see:

- a recursive planning trace
- a flat symbolic plan
- a reactive BT preview
- tick-by-tick execution ending in `SUCCESS`

## Testing

Run the full test suite with:

```powershell
python -m unittest discover -s tests -v
```

The tests cover:

- reactive BT structure
- skip-if-already-satisfied behavior
- dynamic failure recovery
- Scheme 3 revision flow
- recursive-output canonicalization
- contradictory-target validation

## Demo Scenarios

### Dynamic failure recovery

This is the clearest demo of why the tree is reactive.

Run:

```powershell
python -m src.demo_scenarios dynamic_failure
```

What happens:

- the robot picks up the gear
- it starts moving toward the table
- a fault is injected and the gear is dropped
- the next tick re-checks the conditions
- the tree goes back to `Pick(gear)` and recovers

## Real Examples From This Prototype

### 1. Human-in-the-loop correction

In one Scheme 3 run, the model initially planned to insert a gear into a `chassis`. After human feedback, the plan was revised so both the navigation and insertion target changed to `box`, and the updated BT executed successfully.

The interesting part here is not just that the LLM changed the text. The important part is that the tree preview made it easy to inspect whether the revision was internally consistent before execution.

### 2. Behavior Trees as a safety net

In another test, the model produced a plan that effectively tried to re-pick objects after inserting them during an assembly/disassembly-style sequence.

The BT did not pretend that was acceptable. At runtime, it failed safely when it reached:

```text
[Robot] Cannot pick the gear because it is already inserted.
```

That is a good outcome for this prototype. It shows the BT and world-state checks acting as a guardrail against unsafe symbolic planning.

### 3. Recursive planning on a multi-stage task

For a longer instruction involving a screwdriver, hammer, and gear, Scheme 4 decomposed the task into separate primitive subgoals, predicted state between them, and then executed the final flat plan successfully.

That gives the prototype a much closer connection to the recursive planning story described in the paper.

## Limitations

- The symbolic action space is intentionally small.
- The environment is a mock world, not a full robotics stack.
- Scheme 3 works best when the requested revision can still be expressed with `Pick`, `MoveTo`, `Place`, and `Insert`.
- The validator catches important reactive contradictions, but not every possible planning mistake.
- Hugging Face model availability depends on router/provider support and remaining credits.

## What This Project Is Not

This project is not:

- a ROS integration
- a production robotics controller
- a full reproduction of the paper's experimental setup
- a general planner with arbitrary robot skills

It is a focused prototype for showing that the semantic-planning ideas in the paper can be implemented, tested, and analyzed in code.

## Suggested Demo Order

If you are presenting this to someone, a clean order is:

1. run the test suite
2. show the dynamic failure demo
3. run Scheme 3 once and correct a plan interactively
4. run Scheme 4 on the multi-object task

## Final Note

This repo is strongest when it is treated as a clear, inspectable research prototype. It shows the full path from language to symbolic planning to reactive execution, and it makes failure cases visible instead of hiding them.
