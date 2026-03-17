"""
dynamic_crew.py — DynamicCrew multi-agent orchestrator

Architecture
------------
User Query
    └─► Planner Agent (structured LLM call → TaskGraph)
             └─► Execute member agents following TaskGraph topology
                  ├─ Parallel waves  (steps with no unmet deps)
                  └─ Sequential waves (steps whose deps are done)
                       └─► Aggregator Agent
                                └─► Final Output

Usage
-----
.. code-block:: python

    from langchain_openai import ChatOpenAI
    from vinagent.agent import Agent
    from vinagent.multi_agent.dynamic_crew import DynamicCrew

    llm = ChatOpenAI(model="gpt-4o-mini")

    crew = DynamicCrew(
        llm=llm,
        agents={
            "finance_agent": Agent(description="Finance analyst", llm=llm),
            "tech_agent":    Agent(description="Tech analyst",    llm=llm),
            "writer":        Agent(description="Report writer",   llm=llm),
        },
    )

    result = crew.invoke("Write a combined market + tech analysis for Apple Inc.")
    print(result)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Union

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_together import ChatTogether

from vinagent.agent.agent import Agent
from vinagent.task.task import TaskGraph, TaskStep

logger = logging.getLogger(__name__)


class DynamicCrew:
    """
    Dynamic multi-agent orchestrator.

    The crew works in three phases:

    1. **Plan** – a Planner LLM call converts the user query into a
       :class:`~vinagent.task.task.TaskGraph` (structured Pydantic output).

    2. **Execute** – the :class:`TaskGraph` is traversed level-by-level
       (topological sort).  All steps in the same level are independent and
       are executed concurrently via ``asyncio.gather``.

    3. **Aggregate** – an optional Aggregator agent (or the planner LLM itself)
       synthesises all step results into a final answer.

    Parameters
    ----------
    llm : LLM
        Language model used for planning (and aggregation if no ``aggregator``
        is provided).
    agents : Dict[str, Agent]
        Member agents keyed by name.  The Planner will reference these names
        inside ``TaskStep.agent_name``.
    aggregator : Agent, optional
        Dedicated agent to synthesise results.  When *None*, the crew uses
        *llm* directly for aggregation.
    planner_prompt : str, optional
        System prompt injected before the planning call.  A sensible default
        is provided.
    max_iterations : int
        Max iterations forwarded to each member agent's ``invoke`` / ``ainvoke``.
    user_id : str
        User identifier forwarded to agents that persist memory.
    """

    DEFAULT_PLANNER_PROMPT = (
        "You are a master planner for a multi-agent system.\n"
        "Given the user's query and the list of available agents, produce a "
        "structured execution plan (TaskGraph) that breaks the query into discrete "
        "steps.\n\n"
        "Rules:\n"
        "- Each step must be assigned to exactly one agent from the available list.\n"
        "- Use depends_on to express ordering: empty list = can start immediately.\n"
        "- Steps with no shared dependencies will run in PARALLEL.\n"
        "- The input_context field is the literal prompt sent to the agent; use "
        "  {{task_id}} placeholders to inject prior step outputs.\n"
        "- Keep descriptions concise (≤ 2 sentences).\n\n"
        "Available agents:\n{agent_list}\n\n"
        "User query:\n{query}"
    )

    DEFAULT_AGGREGATOR_PROMPT = (
        "You are a synthesis expert. "
        "Given the original user query and the results produced by multiple specialist "
        "agents, write a comprehensive, well-structured final answer.\n\n"
        "Original query:\n{query}\n\n"
        "Agent results:\n{results}\n\n"
        "Final answer:"
    )

    def __init__(
        self,
        llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI],
        agents: Dict[str, Agent],
        aggregator: Optional[Agent] = None,
        planner_prompt: Optional[str] = None,
        max_iterations: int = 5,
        user_id: str = "unknown_user",
    ):
        if not agents:
            raise ValueError("DynamicCrew requires at least one member agent.")

        self.llm = llm
        self.agents = agents
        self.aggregator = aggregator
        self.planner_prompt = planner_prompt or self.DEFAULT_PLANNER_PROMPT
        self.max_iterations = max_iterations
        self.user_id = user_id

        # Build structured-output planner once
        self._structured_planner = self.llm.with_structured_output(
            TaskGraph, method="function_calling"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _agent_list_str(self) -> str:
        """Return a readable list of available agent names for the planner."""
        return "\n".join(f"- {name}" for name in self.agents)

    def _plan(self, query: str) -> TaskGraph:
        """
        Call the Planner LLM synchronously and return a validated TaskGraph.
        """
        prompt = self.planner_prompt.format(
            agent_list=self._agent_list_str(), query=query
        )
        logger.info("DynamicCrew: invoking Planner …")
        task_graph: TaskGraph = self._structured_planner.invoke(prompt)
        self._logging_plan(task_graph)
        return task_graph

    async def _plan_async(self, query: str) -> TaskGraph:
        """Async version of _plan."""
        prompt = self.planner_prompt.format(
            agent_list=self._agent_list_str(), query=query
        )
        logger.info("DynamicCrew: invoking Planner (async) …")
        task_graph: TaskGraph = await self._structured_planner.ainvoke(prompt)
        logger.info(f"DynamicCrew: plan produced {len(task_graph.steps)} step(s).")
        self._logging_plan(task_graph)
        return task_graph

    def _logging_plan(self, plan: TaskGraph):
        """Log the plan."""
        logger.info(f"DynamicCrew: plan produced {len(plan.steps)} step(s).")
        full_content = ""
        for i, step in enumerate(plan.steps):
            content = self._logging_step(step, i, is_show_log=False)
            full_content += content
        logger.info(full_content)
        return full_content

    def _logging_step(
        self, step: TaskStep, step_num: int = 1, is_show_log: bool = True
    ):
        """Log the plan."""
        content = (
            f"\nStep {step_num}: {step.description}\n"
            f"  └─ Task ID: {step.task_id}\n"
            f"  └─ Agent: {step.agent_name}\n"
            f"  └─ Depends On: {step.depends_on}\n"
        )
        if is_show_log:
            logger.info(content)
        return content

    def _run_step_sync(
        self, step: TaskStep, results: Dict[str, str], step_num: int = 1
    ) -> str:
        """Execute a single TaskStep synchronously."""
        agent = self.agents.get(step.agent_name)
        if agent is None:
            raise ValueError(
                f"Step '{step.task_id}' references unknown agent '{step.agent_name}'. "
                f"Available: {list(self.agents)}"
            )
        prompt = self._build_step_prompt(step, results)

        self._logging_step(step, step_num)

        logger.info(
            "\n"
            + "-" * 78
            + "\n"
            + f"DynamicCrew: running step '{step.task_id}' on agent '{step.agent_name}'"
        )
        response = agent.invoke(
            prompt,
            max_iterations=self.max_iterations,
            user_id=self.user_id,
        )
        return response.content if hasattr(response, "content") else str(response)

    async def _run_step_async(
        self, step: TaskStep, results: Dict[str, str], step_num: int = 1
    ) -> str:
        """Execute a single TaskStep asynchronously."""
        agent = self.agents.get(step.agent_name)
        if agent is None:
            raise ValueError(
                f"Step '{step.task_id}' references unknown agent '{step.agent_name}'. "
                f"Available: {list(self.agents)}"
            )
        prompt = self._build_step_prompt(step, results)

        self._logging_step(step, step_num)

        logger.info(
            "\n"
            + "-" * 78
            + "\n"
            + f"DynamicCrew: running step '{step.task_id}' (async) on agent '{step.agent_name}'"
        )
        response = await agent.ainvoke(
            prompt,
            max_iterations=self.max_iterations,
            user_id=self.user_id,
        )
        return response.content if hasattr(response, "content") else str(response)

    def _build_step_prompt(self, step: TaskStep, results: Dict[str, str]) -> str:
        """
        Build the full prompt for a step, injecting prior outputs and the
        expected_output hint.
        """
        # Delegate template rendering to TaskGraph helper
        from vinagent.task.task import TaskGraph as _TG  # avoid circular ref warning

        rendered = step.input_context
        try:
            rendered = step.input_context.format(**results)
        except KeyError as exc:
            logger.warning(
                f"Missing placeholder {exc} in step '{step.task_id}'. Using raw context."
            )
        return f"{rendered}\n\n" f"Expected output: {step.expected_output}"

    def _aggregate_sync(self, query: str, results: Dict[str, str]) -> str:
        """Run the aggregation phase synchronously."""
        results_text = "\n\n".join(f"[{tid}]\n{out}" for tid, out in results.items())
        if self.aggregator:
            prompt = (
                f"Original user query:\n{query}\n\n"
                f"Results from specialist agents:\n{results_text}\n\n"
                f"Please synthesise a comprehensive final answer."
            )
            response = self.aggregator.invoke(
                prompt,
                max_iterations=self.max_iterations,
                user_id=self.user_id,
            )
            return response.content if hasattr(response, "content") else str(response)
        else:
            agg_prompt = self.DEFAULT_AGGREGATOR_PROMPT.format(
                query=query, results=results_text
            )
            response = self.llm.invoke(agg_prompt)
            return response.content if hasattr(response, "content") else str(response)

    async def _aggregate_async(self, query: str, results: Dict[str, str]) -> str:
        """Run the aggregation phase asynchronously."""
        results_text = "\n\n".join(f"[{tid}]\n{out}" for tid, out in results.items())
        if self.aggregator:
            prompt = (
                f"Original user query:\n{query}\n\n"
                f"Results from specialist agents:\n{results_text}\n\n"
                f"Please synthesise a comprehensive final answer."
            )
            response = await self.aggregator.ainvoke(
                prompt,
                max_iterations=self.max_iterations,
                user_id=self.user_id,
            )
            return response.content if hasattr(response, "content") else str(response)
        else:
            agg_prompt = self.DEFAULT_AGGREGATOR_PROMPT.format(
                query=query, results=results_text
            )
            response = await self.llm.ainvoke(agg_prompt)
            return response.content if hasattr(response, "content") else str(response)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def invoke(self, query: str, **kwargs) -> str:
        """
        Synchronously orchestrate the full DynamicCrew pipeline.

        Steps
        -----
        1. Plan   → TaskGraph (structured LLM call)
        2. Execute → level-by-level (parallel steps run sequentially here;
                     for true parallelism use :meth:`ainvoke`)
        3. Aggregate → final answer

        Parameters
        ----------
        query : str
            The user's request.

        Returns
        -------
        str
            The aggregated final answer.
        """
        task_graph = self._plan(query)
        levels = task_graph.topological_levels()

        all_steps = [step for level in levels for step in level]
        step_number = {step.task_id: i for i, step in enumerate(all_steps, start=1)}

        results: Dict[str, str] = {}
        for level_idx, level in enumerate(levels):
            logger.info(
                f"DynamicCrew.invoke: executing level {level_idx + 1}/{len(levels)} "
                f"with {len(level)} step(s): {[s.task_id for s in level]}"
            )
            # In sync mode, run each step sequentially even if they are
            # topologically parallel (use ainvoke for true concurrency)
            for step in level:
                results[step.task_id] = self._run_step_sync(
                    step, results, step_num=step_number[step.task_id]
                )

        logger.info("DynamicCrew.invoke: all steps done, aggregating …")
        return self._aggregate_sync(query, results)

    async def ainvoke(self, query: str, **kwargs) -> str:
        """
        Asynchronously orchestrate the full DynamicCrew pipeline.

        Steps with no unmet dependencies within the same topological level
        are executed **concurrently** via ``asyncio.gather``.

        Parameters
        ----------
        query : str
            The user's request.

        Returns
        -------
        str
            The aggregated final answer.
        """
        task_graph = await self._plan_async(query)
        levels = task_graph.topological_levels()

        all_steps = [step for level in levels for step in level]
        step_number = {step.task_id: i for i, step in enumerate(all_steps, start=1)}

        results: Dict[str, str] = {}
        for level_idx, level in enumerate(levels):
            logger.info(
                f"DynamicCrew.ainvoke: executing level {level_idx + 1}/{len(levels)} "
                f"with {len(level)} parallel step(s): {[s.task_id for s in level]}"
            )
            # All steps in this level are independent — run them concurrently
            step_outputs = await asyncio.gather(
                *[
                    self._run_step_async(
                        step, results, step_num=step_number[step.task_id]
                    )
                    for step in level
                ]
            )
            for step, output in zip(level, step_outputs):
                results[step.task_id] = output

        logger.info("DynamicCrew.ainvoke: all steps done, aggregating …")
        return await self._aggregate_async(query, results)

    def get_plan(self, query: str) -> TaskGraph:
        """
        Return the TaskGraph for *query* without executing it.
        Useful for previewing / debugging the planner's output.
        """
        return self._plan(query)

    async def aget_plan(self, query: str) -> TaskGraph:
        """Async version of :meth:`get_plan`."""
        return await self._plan_async(query)

    def __repr__(self) -> str:
        return (
            f"DynamicCrew(agents={list(self.agents.keys())}, "
            f"aggregator={'yes' if self.aggregator else 'llm-direct'})"
        )
