from typing import Optional
from langchain_core.messages import BaseMessage


def adapter_ai_response_with_tool_calls(
    all_tools: dict,
    response: BaseMessage,
    tool_call: dict,
    tool_call_id: Optional[str] = None,
):
    """Adapt an AgentResponse into an AIMessage with populated tool_calls.

    Args:
        all_tools: The registered tools dict keyed by tool_name.
        response: The AIMessage to mutate with tool_calls.
        tool_call: The tool_call dict from AgentResponse.model_dump().
        tool_call_id: Optional explicit ID to use. When provided this value is
            used as ``tool_calls[0]['id']``; when omitted the static registry
            UUID is used as a fallback. Callers should always pass a fresh UUID
            generated per invocation so that AIMessage and ToolMessage share
            the same ID.
    """
    tool_name = tool_call.get("tool_name")
    if tool_name not in all_tools:
        raise ValueError(
            f"Agent hallucinated non-existent tool: '{tool_name}'. "
            f"Available tools are: {list(all_tools.keys())}"
        )

    selected_tool = all_tools[tool_name]
    adapt_tool = {
        "name": selected_tool["tool_name"],
        "args": tool_call["arguments"],
        "type": "tool_call",
        # Prefer the caller-supplied ID (fresh per invocation) so the
        # AIMessage and ToolMessage always share the exact same ID.
        "id": tool_call_id or selected_tool.get("tool_call_id", "unknown"),
    }
    response.tool_calls = [adapt_tool]
    return response
