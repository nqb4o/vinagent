"""
task.py — DynamicCrew task modelling

`TaskStep`   : a single unit of work in the dynamic plan.
`TaskGraph`  : extends FlowStateGraph to build a dynamic execution graph at
               runtime from a list of `TaskStep` objects produced by the Planner.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TaskStep — one planned unit of work
# ---------------------------------------------------------------------------
class TaskStep(BaseModel):
    """A single step in the planner's task graph.

    Attributes
    ----------
    task_id : str
        Unique identifier for this step (e.g. "step_1", "research", …).
    description : str
        Natural-language description of what this step should accomplish.
    agent_name : str
        Key into ``DynamicCrew.agents`` dict — which agent handles this step.
    depends_on : List[str]
        ``task_id`` values that must complete **before** this step starts.
        Empty list means the step can start immediately (may run in parallel
        with other independent steps).
    input_context : str
        Template for the prompt sent to the agent.
        Use ``{task_id}`` placeholders to reference prior step outputs, e.g.
        ``"Summarise the following research: {research}"``
    expected_output : str
        Description of the required output format / content.
    """

    task_id: str = Field(description="Unique step identifier, e.g. 'step_1'")
    description: str = Field(description="What this step must accomplish")
    agent_name: str = Field(description="Name of the agent that will handle this step")
    depends_on: List[str] = Field(
        default_factory=list,
        description="task_ids that must complete before this step starts",
    )
    input_context: str = Field(
        description=(
            "Prompt template for the agent. Use {task_id} to reference a prior "
            "step's output, e.g. 'Summarise this research: {research}'"
        )
    )
    expected_output: str = Field(
        description="Expected output format or content description"
    )


# ---------------------------------------------------------------------------
# TaskGraph — extends FlowStateGraph for dynamic multi-agent graphs
# ---------------------------------------------------------------------------
class TaskGraph(BaseModel):
    """Structured output returned by the Planner Agent.

    Contains an ordered list of :class:`TaskStep` objects whose
    ``depends_on`` relationships define the directed-acyclic execution graph.
    The ``DynamicCrew`` orchestrator reads this object, builds a concrete
    :class:`~vinagent.graph.operator.FlowStateGraph` at runtime, and executes
    it with the member agents.

    Example
    -------
    A plan with two parallel research steps (A, B) followed by a
    synthesis step (C that depends on both):

    .. code-block:: python

        TaskGraph(steps=[
            TaskStep(
                task_id="research_finance",
                description="Search for recent financial news",
                agent_name="finance_agent",
                depends_on=[],
                input_context="Find the latest financial news about AI companies",
                expected_output="A bullet-point summary of top financial events",
            ),
            TaskStep(
                task_id="research_tech",
                description="Search for recent tech news",
                agent_name="tech_agent",
                depends_on=[],
                input_context="Find the latest tech news about AI companies",
                expected_output="A bullet-point summary of top tech events",
            ),
            TaskStep(
                task_id="synthesis",
                description="Synthesise both research outputs",
                agent_name="synthesis_agent",
                depends_on=["research_finance", "research_tech"],
                input_context=(
                    "Synthesise the following:\n"
                    "Finance research: {research_finance}\n"
                    "Tech research: {research_tech}"
                ),
                expected_output="A comprehensive report combining both domains",
            ),
        ])
    """

    steps: List[TaskStep] = Field(
        description="Ordered list of tasks. depends_on defines the execution DAG."
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_step(self, task_id: str) -> Optional[TaskStep]:
        """Return the step with the given task_id, or None."""
        for step in self.steps:
            if step.task_id == task_id:
                return step
        return None

    def topological_levels(self) -> List[List[TaskStep]]:
        """
        Group steps into levels for execution.

        Steps with no unmet dependencies form the first level and can be
        run in parallel.  Steps whose dependencies are all in previous
        levels form subsequent levels.  This is a simple Kahn-style BFS.

        Returns
        -------
        List[List[TaskStep]]
            Ordered list of *waves*.  All steps in the same wave are
            independent and safe to run concurrently.

        Raises
        ------
        ValueError
            If a dependency references an unknown task_id, or if a cycle
            is detected.
        """
        known_ids = {s.task_id for s in self.steps}
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in known_ids:
                    raise ValueError(
                        f"Step '{step.task_id}' depends on unknown task_id='{dep}'"
                    )

        remaining: Dict[str, TaskStep] = {s.task_id: s for s in self.steps}
        completed: set = set()
        levels: List[List[TaskStep]] = []

        max_iters = len(self.steps) + 1
        for _ in range(max_iters):
            if not remaining:
                break
            ready = [
                s
                for s in remaining.values()
                if all(dep in completed for dep in s.depends_on)
            ]
            if not ready:
                cycle_ids = list(remaining.keys())
                raise ValueError(
                    f"Cycle detected in TaskGraph among steps: {cycle_ids}"
                )
            levels.append(ready)
            for s in ready:
                completed.add(s.task_id)
                del remaining[s.task_id]

        return levels

    def build_prompt(self, step: TaskStep, results: Dict[str, str]) -> str:
        """
        Render the ``input_context`` template for *step*, substituting
        prior step outputs stored in *results*.

        Parameters
        ----------
        step : TaskStep
            The step whose prompt to build.
        results : Dict[str, str]
            Mapping of ``task_id → output`` for all already-completed steps.

        Returns
        -------
        str
            The fully-rendered prompt string ready to send to the agent.
        """
        try:
            return step.input_context.format(**results)
        except KeyError as exc:
            logger.warning(
                f"TaskGraph.build_prompt: missing key {exc} for step '{step.task_id}'. "
                "Falling back to raw input_context."
            )
            return step.input_context
