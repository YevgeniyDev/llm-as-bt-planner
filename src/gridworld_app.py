"""
Tkinter-based visual tester for the LLM-driven gridworld harness.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, Optional

from dotenv import load_dotenv
import py_trees

from .gridworld_domain import default_render_shape
from .gridworld_env import (
    DEFAULT_TYPED_SCENARIO,
    SimulationResult,
    TypedGridWorldEnv,
    build_env_from_typed_scenario,
    resolve_typed_scenario_text,
)
from .gridworld_layouts import layout_registry
from .gridworld_presets import (
    CUSTOM_GRIDWORLD_SCENARIO_PLACEHOLDER,
    GRIDWORLD_PRESETS,
    GridWorldPreset,
    resolve_preset_payload,
)
from .llm_client import LLMTaskPlanner


CELL_PADDING = 6
ROBOT_COLORS = ("#ff5a5f", "#ffd166", "#06d6a0", "#4cc9f0", "#f72585", "#f77f00")
OBJECT_COLORS = {
    "circle": "#f1fa3c",
    "square": "#4ade80",
    "triangle": "#ff6b6b",
    "gear": "#f59e0b",
    "shaft": "#60a5fa",
    "pin": "#c084fc",
    "plate": "#34d399",
}
FIXED_LOCATION_COLORS = {
    "tool_station": "#f59e0b",
    "fixture": "#22d3ee",
}


def resolve_gridworld_app_scenario_text(raw_text: str) -> str:
    cleaned_text = raw_text.strip()
    if cleaned_text == CUSTOM_GRIDWORLD_SCENARIO_PLACEHOLDER:
        return DEFAULT_TYPED_SCENARIO
    return resolve_typed_scenario_text(cleaned_text)


def capture_text_scroll_state(text_widget: tk.Text) -> Dict[str, float | bool]:
    y_first, y_last = text_widget.yview()
    x_first, _x_last = text_widget.xview()
    return {
        "y_first": y_first,
        "x_first": x_first,
        "stick_bottom": y_last >= 0.999,
    }


class GridWorldTesterApp:
    def __init__(
        self,
        scenario_text: Optional[str] = None,
        num_robots: int = 3,
        num_circles: Optional[int] = None,
        layout_name: str = "open_room",
        seed: int = 0,
        max_steps: int = 70,
    ) -> None:
        load_dotenv()
        self.root = tk.Tk()
        self.root.title("MRBTP GridWorld Tester")
        self.root.geometry("1360x900")
        self.root.minsize(1100, 760)

        self.layout_names = sorted(layout_registry())
        self.presets = {preset.name: preset for preset in GRIDWORLD_PRESETS}
        self.last_result: Optional[SimulationResult] = None
        self.current_env: Optional[TypedGridWorldEnv] = None
        self.current_tree_preview = ""
        self.current_provider = ""
        self.current_model = ""
        self.current_frame_index = 0
        self.completion_popup_shown = False
        self.play_job: Optional[str] = None
        self.is_playing = False
        self.is_running_scenario = False
        self.section_frames: Dict[str, ttk.LabelFrame] = {}
        self.section_text_widgets: Dict[str, tk.Text] = {}
        self.section_visibility_vars: Dict[str, tk.BooleanVar] = {}
        self.section_order = ("summary", "events", "state", "plan", "bt")
        self.section_titles = {
            "summary": "Scenario Summary",
            "events": "Current Actions",
            "state": "Robots And Objects",
            "plan": "Derived Plan",
            "bt": "MRBTP BT Preview",
        }
        self.section_min_sizes = {
            "summary": 120,
            "events": 120,
            "state": 140,
            "plan": 140,
            "bt": 140,
        }
        self.sections_pane: Optional[tk.PanedWindow] = None

        self.status_var = tk.StringVar(value="Ready.")
        self.tick_var = tk.StringVar(value="Tick: -")
        self.phase_var = tk.StringVar(value="Phase: -")
        self.result_var = tk.StringVar(value="Result: -")
        self.layout_var = tk.StringVar(value=layout_name if layout_name in self.layout_names else self.layout_names[0])
        self.num_robots_var = tk.IntVar(value=max(1, num_robots))
        self.num_circles_var = tk.IntVar(value=max(1, num_circles if num_circles is not None else num_robots))
        self.seed_var = tk.IntVar(value=seed)
        self.max_steps_var = tk.IntVar(value=max(1, max_steps))
        self.preset_var = tk.StringVar(value=GRIDWORLD_PRESETS[0].name if GRIDWORLD_PRESETS else "")
        if scenario_text and scenario_text.strip():
            initial_scenario_text = scenario_text.strip()
        elif GRIDWORLD_PRESETS:
            initial_scenario_text = GRIDWORLD_PRESETS[0].scenario_text
        else:
            initial_scenario_text = DEFAULT_TYPED_SCENARIO
        self.scenario_var = tk.StringVar(value=initial_scenario_text)

        self._build_ui()
        self._clear_loaded_scenario(status_message="Ready.")

    def run(self) -> Optional[bool]:
        self.root.mainloop()
        if self.last_result is None:
            return None
        return self.last_result.completed

    def _build_ui(self) -> None:
        root_frame = ttk.Frame(self.root, padding=10)
        root_frame.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(root_frame, text="Scenario Controls", padding=10)
        controls.pack(fill=tk.X, expand=False)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(8, weight=1)

        ttk.Label(controls, text="Preset").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=(0, 6))
        preset_box = ttk.Combobox(
            controls,
            textvariable=self.preset_var,
            values=list(self.presets),
            state="readonly",
            width=28,
        )
        preset_box.grid(row=0, column=1, sticky="w", padx=(0, 12), pady=(0, 6))
        preset_box.bind("<<ComboboxSelected>>", self._on_preset_selected)

        ttk.Label(controls, text="Layout").grid(row=0, column=2, sticky="w", padx=(0, 6), pady=(0, 6))
        ttk.Combobox(
            controls,
            textvariable=self.layout_var,
            values=self.layout_names,
            state="readonly",
            width=16,
        ).grid(row=0, column=3, sticky="w", padx=(0, 12), pady=(0, 6))

        ttk.Label(controls, text="Robots").grid(row=0, column=4, sticky="w", padx=(0, 6), pady=(0, 6))
        ttk.Spinbox(controls, from_=1, to=8, textvariable=self.num_robots_var, width=6).grid(
            row=0, column=5, sticky="w", padx=(0, 12), pady=(0, 6)
        )

        ttk.Label(controls, text="Objects").grid(row=0, column=6, sticky="w", padx=(0, 6), pady=(0, 6))
        ttk.Spinbox(controls, from_=1, to=12, textvariable=self.num_circles_var, width=6).grid(
            row=0, column=7, sticky="w", padx=(0, 12), pady=(0, 6)
        )

        ttk.Label(controls, text="Seed").grid(row=0, column=8, sticky="e", padx=(0, 6), pady=(0, 6))
        ttk.Spinbox(controls, from_=0, to=9999, textvariable=self.seed_var, width=8).grid(
            row=0, column=9, sticky="w", padx=(0, 12), pady=(0, 6)
        )

        ttk.Label(controls, text="Max Steps").grid(row=0, column=10, sticky="e", padx=(0, 6), pady=(0, 6))
        ttk.Spinbox(controls, from_=5, to=200, textvariable=self.max_steps_var, width=8).grid(
            row=0, column=11, sticky="w", padx=(0, 12), pady=(0, 6)
        )

        self.run_button = ttk.Button(controls, text="Run Scenario", command=self._run_scenario)
        self.run_button.grid(
            row=0,
            column=12,
            sticky="e",
            padx=(6, 0),
            pady=(0, 6),
        )

        ttk.Label(controls, text="Instruction").grid(row=1, column=0, sticky="nw", padx=(0, 6))
        self.scenario_text = tk.Text(controls, height=4, wrap=tk.WORD)
        self.scenario_text.grid(row=1, column=1, columnspan=12, sticky="nsew")
        self.scenario_text.insert("1.0", self.scenario_var.get())

        playback = ttk.Frame(root_frame, padding=(0, 10, 0, 10))
        playback.pack(fill=tk.X, expand=False)
        playback.columnconfigure(6, weight=1)
        ttk.Button(playback, text="Play", command=self._play).grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Button(playback, text="Pause", command=self._pause).grid(row=0, column=1, sticky="w", padx=(0, 6))
        ttk.Button(playback, text="Step", command=self._step_forward).grid(row=0, column=2, sticky="w", padx=(0, 12))
        ttk.Label(playback, textvariable=self.tick_var).grid(row=0, column=3, sticky="w", padx=(0, 12))
        ttk.Label(playback, textvariable=self.phase_var).grid(row=0, column=4, sticky="w", padx=(0, 12))
        ttk.Label(playback, textvariable=self.result_var).grid(row=0, column=5, sticky="w", padx=(0, 12))
        self.status_label = ttk.Label(playback, textvariable=self.status_var, anchor="e")
        self.status_label.grid(row=0, column=7, sticky="e")

        body = ttk.Panedwindow(root_frame, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(body)
        right_panel = ttk.Frame(body)
        body.add(left_panel, weight=3)
        body.add(right_panel, weight=2)

        self.canvas = tk.Canvas(left_panel, background="#1f2937", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._render_current_frame())

        right_panel.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)

        section_controls = ttk.LabelFrame(right_panel, text="Panels", padding=8)
        section_controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for column_index, section_key in enumerate(self.section_order):
            self.section_visibility_vars[section_key] = tk.BooleanVar(
                value=section_key != "summary"
            )
            ttk.Checkbutton(
                section_controls,
                text=self.section_titles[section_key].replace("Scenario ", "").replace("Robots And ", ""),
                variable=self.section_visibility_vars[section_key],
                command=self._refresh_section_visibility,
            ).grid(row=0, column=column_index, sticky="w", padx=(0, 8))

        self.sections_pane = tk.PanedWindow(
            right_panel,
            orient=tk.VERTICAL,
            sashrelief=tk.RAISED,
            sashwidth=6,
            sashcursor="sb_v_double_arrow",
            showhandle=True,
            handlesize=10,
            handlepad=4,
            opaqueresize=True,
            bd=0,
        )
        self.sections_pane.grid(row=1, column=0, sticky="nsew")

        self.summary_text = self._create_text_section(
            section_key="summary",
            parent=self.sections_pane,
            wrap=tk.WORD,
            height=7,
        )
        self.events_text = self._create_text_section(
            section_key="events",
            parent=self.sections_pane,
            wrap=tk.WORD,
            height=7,
        )
        self.state_text = self._create_text_section(
            section_key="state",
            parent=self.sections_pane,
            wrap=tk.WORD,
            height=8,
        )
        self.plan_text = self._create_text_section(
            section_key="plan",
            parent=self.sections_pane,
            wrap=tk.NONE,
            height=8,
            horizontal_scroll=True,
        )
        self.bt_text = self._create_text_section(
            section_key="bt",
            parent=self.sections_pane,
            wrap=tk.NONE,
            height=8,
            horizontal_scroll=True,
        )
        self._refresh_section_visibility()
        self.root.after_idle(self._reset_section_sashes)

    def _populate_summary_placeholder(self) -> None:
        self._set_text_widget_content(
            self.summary_text,
            "Run a scenario to see the scenario summary.",
        )

    def _clear_loaded_scenario(self, status_message: str = "Ready.") -> None:
        self.last_result = None
        self.current_env = None
        self.current_frame_index = 0
        self.completion_popup_shown = False
        self.current_tree_preview = ""
        self.current_provider = ""
        self.current_model = ""
        self.tick_var.set("Tick: -")
        self.phase_var.set("Phase: -")
        self.result_var.set("Result: -")
        self.status_var.set(status_message)
        self._populate_summary_placeholder()
        self._populate_plan_text("", "")
        self._populate_state_text("", "")
        self._update_event_panel([])
        self._draw_placeholder()

    def _on_preset_selected(self, _event=None) -> None:
        preset = self.presets.get(self.preset_var.get())
        if preset is None:
            return

        self.layout_var.set(preset.layout_name)
        self.num_robots_var.set(preset.num_robots)
        self.num_circles_var.set(preset.num_circles)
        self.scenario_text.delete("1.0", tk.END)
        self.scenario_text.insert("1.0", preset.scenario_text)

    def _create_text_section(
        self,
        section_key: str,
        parent: tk.PanedWindow,
        wrap: str,
        height: int,
        horizontal_scroll: bool = False,
    ) -> tk.Text:
        section_frame = ttk.LabelFrame(parent, text=self.section_titles[section_key], padding=8)
        container = ttk.Frame(section_frame)
        container.pack(fill=tk.BOTH, expand=True)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        text_widget = tk.Text(container, height=height, wrap=wrap)
        text_widget.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(container, orient=tk.VERTICAL, command=text_widget.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        text_widget.configure(yscrollcommand=y_scroll.set)

        if horizontal_scroll:
            x_scroll = ttk.Scrollbar(container, orient=tk.HORIZONTAL, command=text_widget.xview)
            x_scroll.grid(row=1, column=0, sticky="ew")
            text_widget.configure(xscrollcommand=x_scroll.set)

        self.section_frames[section_key] = section_frame
        self.section_text_widgets[section_key] = text_widget
        return text_widget

    def _refresh_section_visibility(self) -> None:
        if self.sections_pane is None:
            return

        for section_key in self.section_order:
            section_frame = self.section_frames.get(section_key)
            if section_frame is None:
                continue
            try:
                self.sections_pane.forget(section_frame)
            except tk.TclError:
                pass

        visible_count = 0
        for section_key in self.section_order:
            if not self.section_visibility_vars[section_key].get():
                continue
            section_frame = self.section_frames[section_key]
            self.sections_pane.add(
                section_frame,
                minsize=self.section_min_sizes[section_key],
                stretch="always",
            )
            visible_count += 1

        if visible_count:
            self.root.after_idle(self._reset_section_sashes)

    def _reset_section_sashes(self) -> None:
        if self.sections_pane is None:
            return

        pane_widgets = self.sections_pane.panes()
        if len(pane_widgets) < 2:
            return

        self.sections_pane.update_idletasks()
        total_height = max(self.sections_pane.winfo_height(), 1)
        visible_frames = [
            self.root.nametowidget(widget_name)
            for widget_name in pane_widgets
        ]
        min_sizes = []
        for frame in visible_frames:
            section_key = next(
                (key for key, value in self.section_frames.items() if value == frame),
                None,
            )
            min_sizes.append(self.section_min_sizes.get(section_key or "", 100))

        sash_target = 0
        remaining_height = total_height
        remaining_panes = len(visible_frames)
        for sash_index in range(len(visible_frames) - 1):
            min_remaining = sum(min_sizes[sash_index + 1 :])
            suggested = total_height * (sash_index + 1) / len(visible_frames)
            sash_target = max(sash_target + min_sizes[sash_index], int(suggested))
            sash_target = min(sash_target, total_height - min_remaining)
            self.sections_pane.sash_place(sash_index, 0, sash_target)
            remaining_height -= sash_target
            remaining_panes -= 1

    def _set_text_widget_content(self, text_widget: tk.Text, content: str) -> None:
        scroll_state = capture_text_scroll_state(text_widget)
        text_widget.configure(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.insert("1.0", content)
        text_widget.configure(state=tk.DISABLED)

        if scroll_state["stick_bottom"]:
            text_widget.yview_moveto(1.0)
        else:
            text_widget.yview_moveto(float(scroll_state["y_first"]))
        text_widget.xview_moveto(float(scroll_state["x_first"]))

    def _run_scenario(self) -> None:
        if self.is_running_scenario:
            return

        self._pause()
        scenario_text = resolve_gridworld_app_scenario_text(self.scenario_text.get("1.0", tk.END))
        requested_robots = max(1, self.num_robots_var.get())
        requested_objects = max(1, self.num_circles_var.get())
        selected_layout = self.layout_var.get()
        scenario_payload = resolve_preset_payload(
            preset_name=self.preset_var.get(),
            scenario_text=scenario_text,
            num_robots=requested_robots,
            num_objects=requested_objects,
            layout_name=selected_layout,
        )
        self.is_running_scenario = True
        self.run_button.state(["disabled"])
        self.status_var.set(
            "Loading built-in scenario..."
            if scenario_payload is not None
            else "Planning scenario with the LLM..."
        )
        self.root.update_idletasks()

        try:
            planner = None if scenario_payload is not None else LLMTaskPlanner()
            env = build_env_from_typed_scenario(
                scenario_text=scenario_text,
                num_robots=requested_robots,
                num_circles=requested_objects,
                layout_name=selected_layout,
                seed=max(0, self.seed_var.get()),
                planner=planner,
                scenario_payload=scenario_payload,
            )
            tree = env.build_behavior_tree()
            result = env.run(max_steps=max(1, self.max_steps_var.get()))
        except Exception as error:  # pragma: no cover - GUI path
            self._clear_loaded_scenario(status_message="Scenario failed to build.")
            messagebox.showerror("GridWorld Error", str(error))
            return
        finally:
            self.is_running_scenario = False
            self.run_button.state(["!disabled"])

        self.current_env = env
        self.last_result = result
        self.current_frame_index = 0
        self.completion_popup_shown = False
        self.current_tree_preview = py_trees.display.unicode_tree(tree.root, show_status=False)
        if planner is None:
            self.current_provider = "built-in"
            self.current_model = "preset"
        else:
            self.current_provider = planner.provider
            self.current_model = planner.model
        self.result_var.set("Result: {}".format("SUCCESS" if result.completed else "INCOMPLETE"))
        self.status_var.set(
            "Loaded {} frames using {}.".format(
                len(result.history),
                "built-in preset"
                if scenario_payload is not None
                else "{} / {}".format(planner.provider, planner.model),
            )
        )

        self._populate_summary_text(env, result)
        self._populate_plan_text(env.build_symbolic_plan(), self.current_tree_preview)
        self._render_current_frame()
        self._play()

    def _populate_summary_text(self, env: TypedGridWorldEnv, result: SimulationResult) -> None:
        summary_lines = [
            "Instruction:",
            result.scenario.raw_text,
            "",
            "LLM summary:",
            result.scenario.task_summary,
            "",
            "Provider / model: {} / {}".format(self.current_provider, self.current_model),
            "Layout: {}".format(env.layout.name),
            "Robots: {}".format(", ".join(robot.name for robot in result.scenario.robots)),
            "Goals: {}".format(
                ", ".join(
                    "{} -> {}".format(goal.object_name, goal.target)
                    for goal in result.scenario.success_conditions
                )
            ),
        ]
        self._set_text_widget_content(self.summary_text, "\n".join(summary_lines))

    def _populate_plan_text(self, plan, bt_preview: str) -> None:
        plan_block = ""
        if plan:
            plan_lines = []
            for step_index, step in enumerate(plan, start=1):
                robot_name = step.get("robot")
                action_name = step.get("action")
                detail = ", ".join(
                    "{}={}".format(key, value)
                    for key, value in step.items()
                    if key not in {"robot", "action"}
                )
                if detail:
                    plan_lines.append("{}. {} {} ({})".format(step_index, robot_name, action_name, detail))
                else:
                    plan_lines.append("{}. {} {}".format(step_index, robot_name, action_name))
            plan_block = "\n".join(plan_lines)

        self._set_text_widget_content(
            self.plan_text,
            plan_block or "Run a scenario to see the derived plan.",
        )
        self._set_text_widget_content(
            self.bt_text,
            bt_preview or "Run a scenario to see the BT preview.",
        )

    def _populate_state_text(self, robot_state_text: str, object_state_text: str) -> None:
        combined_text = "\n".join(
            line for line in [robot_state_text, "", object_state_text] if line is not None
        ).strip()
        self._set_text_widget_content(self.state_text, combined_text or "No frame loaded yet.")

    def _render_current_frame(self) -> None:
        if self.last_result is None or self.current_env is None:
            self._draw_placeholder()
            return

        snapshot = self.last_result.history[self.current_frame_index]
        frame = snapshot.frame
        self.tick_var.set("Tick: {}/{}".format(snapshot.tick, self.last_result.steps_run))
        self.phase_var.set(
            "Phase: {}/{}".format(
                min(frame.current_phase_index + 1, max(frame.total_phases, 1)),
                max(frame.total_phases, 1),
            )
        )
        self._draw_frame(frame)
        self._update_event_panel(snapshot.events)
        self._update_state_panel(frame)
        self._show_completion_popup_if_needed()

    def _show_completion_popup_if_needed(self) -> None:
        if self.last_result is None or self.completion_popup_shown:
            return
        if self.current_frame_index != len(self.last_result.history) - 1:
            return

        self.completion_popup_shown = True
        if self.last_result.completed:
            messagebox.showinfo(
                "GridWorld Result",
                "Scenario finished successfully in {} steps.".format(
                    self.last_result.steps_run
                ),
            )
            return

        messagebox.showwarning(
            "GridWorld Result",
            "Scenario finished without reaching all goals after {} steps.".format(
                self.last_result.steps_run
            ),
        )

    def _update_event_panel(self, events) -> None:
        if events:
            event_text = "\n".join(events)
        else:
            event_text = "No actions yet."
        self._set_text_widget_content(self.events_text, event_text)

    def _update_state_panel(self, frame) -> None:
        robot_lines = ["Robots:"]
        for robot in frame.robots:
            robot_lines.append(
                "- {} at {} role={} carrying={} tool={}".format(
                    robot.name,
                    robot.position,
                    robot.role,
                    robot.carrying or "nothing",
                    robot.current_tool,
                )
            )

        object_lines = ["Objects:"]
        for obj in frame.objects:
            location = obj.held_by or obj.position
            object_lines.append(
                "- {} kind={} shape={} at {} delivered={} inserted={}".format(
                    obj.name,
                    obj.kind,
                    obj.shape,
                    location,
                    obj.delivered,
                    obj.inserted_target or "no",
                )
            )

        fixed_lines = ["", "Fixed locations:"]
        for location in frame.fixed_locations:
            fixed_lines.append(
                "- {} type={} at {}".format(
                    location.name,
                    location.category,
                    location.position,
                )
            )

        self._populate_state_text("\n".join(robot_lines), "\n".join(object_lines + fixed_lines))

    def _draw_placeholder(self) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 400)
        height = max(self.canvas.winfo_height(), 400)
        self.canvas.create_rectangle(0, 0, width, height, fill="#111827", outline="")
        self.canvas.create_text(
            width / 2,
            height / 2,
            text="Run a scenario to open the visual gridworld tester.",
            fill="#f9fafb",
            font=("Segoe UI", 16, "bold"),
        )

    def _draw_frame(self, frame) -> None:
        self.canvas.delete("all")
        canvas_width = max(self.canvas.winfo_width(), 500)
        canvas_height = max(self.canvas.winfo_height(), 500)
        self.canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="#111827", outline="")

        cell_size = min((canvas_width - 40) / frame.width, (canvas_height - 40) / frame.height)
        cell_size = max(cell_size, 18)
        origin_x = (canvas_width - frame.width * cell_size) / 2
        origin_y = (canvas_height - frame.height * cell_size) / 2

        for y in range(frame.height):
            for x in range(frame.width):
                left = origin_x + x * cell_size
                top = origin_y + y * cell_size
                right = left + cell_size
                bottom = top + cell_size
                fill_color = "#0f172a"
                if (x, y) in frame.walls:
                    fill_color = "#6b7280"
                self.canvas.create_rectangle(
                    left,
                    top,
                    right,
                    bottom,
                    fill=fill_color,
                    outline="#374151",
                )

        for location in frame.fixed_locations:
            self._draw_fixed_location(location, origin_x, origin_y, cell_size)

        for goal_index, (object_name, target_name, position) in enumerate(frame.goals):
            color = ROBOT_COLORS[goal_index % len(ROBOT_COLORS)]
            left, top, right, bottom = self._cell_bounds(position, origin_x, origin_y, cell_size)
            self.canvas.create_rectangle(
                left + 4,
                top + 4,
                right - 4,
                bottom - 4,
                outline=color,
                width=3,
            )
            self.canvas.create_text(
                (left + right) / 2,
                bottom - 10,
                text=target_name,
                fill="#d1d5db",
                font=("Segoe UI", 8),
            )

        free_objects = [obj for obj in frame.objects if obj.held_by is None]
        for object_state in free_objects:
            self._draw_object(object_state, origin_x, origin_y, cell_size)

        robots_by_position: Dict[tuple, list] = {}
        for robot in frame.robots:
            robots_by_position.setdefault(robot.position, []).append(robot)

        for robots in robots_by_position.values():
            for index, robot in enumerate(robots):
                self._draw_robot(robot, index, len(robots), origin_x, origin_y, cell_size)

    def _draw_object(self, object_state, origin_x: float, origin_y: float, cell_size: float) -> None:
        left, top, right, bottom = self._cell_bounds(object_state.position, origin_x, origin_y, cell_size)
        color = OBJECT_COLORS.get(object_state.kind, OBJECT_COLORS.get(object_state.shape, "#f9fafb"))
        inset = CELL_PADDING + 4
        render_shape = default_render_shape(object_state.kind)
        if object_state.shape in {"circle", "square", "triangle"}:
            render_shape = object_state.shape

        if render_shape == "circle":
            self.canvas.create_oval(left + inset, top + inset, right - inset, bottom - inset, fill=color, outline="")
        elif render_shape == "square":
            self.canvas.create_rectangle(left + inset, top + inset, right - inset, bottom - inset, fill=color, outline="")
        else:
            self.canvas.create_polygon(
                (left + right) / 2,
                top + inset,
                right - inset,
                bottom - inset,
                left + inset,
                bottom - inset,
                fill=color,
                outline="",
            )

        self.canvas.create_text(
            (left + right) / 2,
            bottom - 8,
            text=object_state.name,
            fill="#f8fafc",
            font=("Segoe UI", 8, "bold"),
        )

    def _draw_fixed_location(self, location, origin_x: float, origin_y: float, cell_size: float) -> None:
        left, top, right, bottom = self._cell_bounds(location.position, origin_x, origin_y, cell_size)
        color = FIXED_LOCATION_COLORS.get(location.category, "#d1d5db")
        inset = 8

        self.canvas.create_rectangle(
            left + inset,
            top + inset,
            right - inset,
            bottom - inset,
            outline=color,
            width=2,
            dash=(4, 2),
        )
        self.canvas.create_text(
            (left + right) / 2,
            top + 11,
            text=location.name,
            fill=color,
            font=("Segoe UI", 7, "bold"),
        )

    def _draw_robot(
        self,
        robot,
        stack_index: int,
        stack_size: int,
        origin_x: float,
        origin_y: float,
        cell_size: float,
    ) -> None:
        left, top, right, bottom = self._cell_bounds(robot.position, origin_x, origin_y, cell_size)
        color_index = max(0, int(robot.name.split("_")[-1]) - 1) if robot.name.split("_")[-1].isdigit() else stack_index
        color = ROBOT_COLORS[color_index % len(ROBOT_COLORS)]

        offset_step = min(10, cell_size / 5)
        offset_x = (stack_index - (stack_size - 1) / 2) * offset_step
        offset_y = -(stack_index % 2) * offset_step / 2
        center_x = (left + right) / 2 + offset_x
        center_y = (top + bottom) / 2 + offset_y
        size = cell_size / 3

        self.canvas.create_polygon(
            center_x,
            center_y - size,
            center_x + size,
            center_y + size,
            center_x - size,
            center_y + size,
            fill=color,
            outline="#111827",
            width=1.5,
        )
        self.canvas.create_text(
            center_x,
            center_y + size + 10,
            text=robot.name,
            fill="#f8fafc",
            font=("Segoe UI", 8, "bold"),
        )

    def _cell_bounds(self, position, origin_x: float, origin_y: float, cell_size: float):
        x, y = position
        left = origin_x + x * cell_size
        top = origin_y + y * cell_size
        return left, top, left + cell_size, top + cell_size

    def _play(self) -> None:
        if self.last_result is None:
            return
        self.is_playing = True
        self._schedule_next_frame()

    def _pause(self) -> None:
        self.is_playing = False
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
            self.play_job = None

    def _schedule_next_frame(self) -> None:
        if not self.is_playing or self.last_result is None:
            return
        self.play_job = self.root.after(500, self._advance_frame)

    def _advance_frame(self) -> None:
        self.play_job = None
        if self.last_result is None:
            return

        if self.current_frame_index < len(self.last_result.history) - 1:
            self.current_frame_index += 1
            self._render_current_frame()
            if self.current_frame_index < len(self.last_result.history) - 1:
                self._schedule_next_frame()
            else:
                self.is_playing = False
            return

        self.is_playing = False

    def _step_forward(self) -> None:
        self._pause()
        if self.last_result is None:
            return
        if self.current_frame_index < len(self.last_result.history) - 1:
            self.current_frame_index += 1
            self._render_current_frame()


def launch_gridworld_tester(
    scenario_text: Optional[str] = None,
    num_robots: int = 3,
    num_circles: Optional[int] = None,
    layout_name: str = "open_room",
    seed: int = 0,
    max_steps: int = 70,
) -> Optional[bool]:
    app = GridWorldTesterApp(
        scenario_text=scenario_text,
        num_robots=num_robots,
        num_circles=num_circles,
        layout_name=layout_name,
        seed=seed,
        max_steps=max_steps,
    )
    return app.run()
