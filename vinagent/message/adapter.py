from langchain_core.messages.base import BaseMessage


def adapter_ai_response_with_tool_calls(
    all_tools: dict, response: BaseMessage, tool_call: dict
):
    tool_name = tool_call.get("tool_name")
    if tool_name not in all_tools:
        raise ValueError(
            f"Agent hallucinated non-existent tool: '{tool_name}'. "
            f"Available tools are: {list(all_tools.keys())}"
        )

    adapt_tool = {}
    selected_tool = all_tools[tool_name]
    adapt_tool["name"] = selected_tool["tool_name"]
    adapt_tool["args"] = tool_call["arguments"]
    adapt_tool["type"] = "tool_call"
    adapt_tool["id"] = selected_tool.get("tool_call_id", "unknown")
    response.tool_calls = [adapt_tool]
    return response
