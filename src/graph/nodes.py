from enum import Enum
from typing import List, Dict, Any
from datetime import datetime
import json
import logging
import time
from src.config import TOP_K, OPENAI_MODEL, MAX_TOOL_HOPS, TEMPERATURE, MAX_NODE_ITERATIONS
from src.graph.state import State, ToolCall, build_initial_state
from src.graph.registry import load_index_and_chunks, load_tools
from src.rag.retriever import retrieve
from src.guards.policy import apply_guards, mask_pii, check_grounding_required
from src.observability.telemetry import log_react_step, log_tool_call, log_node_entry, log_node_exit
from openai import OpenAI
client = OpenAI()


class NodeName(str, Enum):
    RAG = "rag"
    FINALIZE = "finalize"
    TOOL = "tool"


def initialize_node(state: State) -> State:
    """Initialize the state."""
    state = build_initial_state(state["query"])
    return state


def guard_node(state: State) -> State:
    """Check guards and set refusal message if query should be refused."""
    query = state["query"]

    passed, refusal_msg, masked_query = apply_guards(
        query, retrieved_chunks=None)
    if not passed:
        logging.warning(f"Query refused by guard: {refusal_msg}")
        state["final_answer"] = refusal_msg
        if masked_query and masked_query != query:
            state["query"] = masked_query
    return state


def generate_tool_call_answer(query: str, tool_calls: List[ToolCall]) -> str:
    """Generate a response using the tool calls."""
    logging.debug(f"Tool calls for answer generation: {tool_calls}")
    prompt = f"""You are given a question and a list of tool calls.
    You need to use the tool calls to answer the question.
    Tool calls: {tool_calls}
    Question: {query}
    Answer: """
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based ONLY on the provided tools."},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def tool_node(state: State) -> State:
    log_node_entry("tool", state)
    tools, tool_definitions = load_tools()
    query = state["query"]
    tool_names = ", ".join(list(tools.keys()))

    # ReAct-style prompt
    system = (
        "Answer the following questions as best you can. You have access to the following tools:\n\n"
        f"{tools}\n\nUse the following format:\n\n"
        "Question: the input question you must answer\n"
        "Thought: you should always think about what to do\n"
        f"Action: the action to take, should be one of [{tool_names}]\n"
        "Action Input: the input to the action\n"
        "Observation: the result of the action\n"
        "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
        "Thought: I now know the final answer\n"
        "Final Answer: the final answer to the original input question\n\n"
        "**IMPORTANT:**\n"
        "- After a tool reports success (e.g., 'Successfully sent message'), stop, don't repeat actions, and summarize the result in your Final Answer.\n"
        "- If you already retrieved information from a tool, use that information to answer. Do not call the same tool again with the same arguments.\n"
        "- Once you have the information you need, provide your Final Answer immediately.\n\n"
        "Begin!"
    )

    agent_scratchpad = ""
    messages = [
        {"role": "system", "content": system.replace("{tools}", tool_names)},
    ]

    final_answer: str | None = None
    iteration = 0
    for iteration in range(MAX_TOOL_HOPS):
        user_content = (
            f"Question: {query}\nThought:{agent_scratchpad}"
        )
        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=tool_definitions,
            tool_choice="required",
            temperature=TEMPERATURE,
        )

        choice = response.choices[0].message
        tool_calls = choice.tool_calls

        if not tool_calls:
            logging.warning("No tool call selected by planner")
            break

        messages.append(choice)

        tool_results = []
        has_duplicate = False
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            for prev_call in state.get("tool_calls", []):
                if prev_call["name"] == tool_name and prev_call.get("arguments") == tool_args:
                    logging.warning(
                        f"Duplicate tool call detected: {tool_name}({tool_args}). Stopping loop.")
                    has_duplicate = True
                    break

            if has_duplicate:
                break

            log_react_step("action", f"{tool_name}({tool_args})", {
                           "iteration": iteration})

            tool_fn = tools[tool_name]
            try:
                result = tool_fn(**tool_args)

                state["tool_calls"].append(
                    ToolCall(name=tool_name, arguments=tool_args, result=result, timestamp=datetime.now()))

                result = str(result)[:500]
                log_react_step("observation", result, {
                               "tool": tool_name, "iteration": iteration})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                logging.error(f"Tool call {tool_name} failed: {error_msg}")
                log_react_step("observation", error_msg, {
                               "tool": tool_name, "error": True, "iteration": iteration})
                result = error_msg
                state["tool_calls"].append(
                    ToolCall(name=tool_name, arguments=tool_args, result=error_msg, timestamp=datetime.now()))

            tool_results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_name,
                "content": result
            })

            # Update scratchpad with Observation and a new Thought
            result_str = str(result)[:500]
            agent_scratchpad += (
                f"\nAction: {tool_name}\nAction Input: {tool_args}\nObservation: {result_str}\nThought:"
            )

        if has_duplicate:
            final_answer = generate_tool_call_answer(
                query, state["tool_calls"])
            break

        messages.extend(tool_results)

    if not final_answer:
        final_answer = generate_tool_call_answer(query, state["tool_calls"])

    state["final_answer"] = final_answer
    state["iteration_count"] += 1
    log_node_exit("tool", state)
    return state


def finalize_node(state: State) -> State:
    """Format and finalize the response."""
    # Mask PII in final answer before returning
    if state.get("final_answer"):
        masked_answer = mask_pii(state["final_answer"])
        state["final_answer"] = masked_answer
        state["messages"].append({
            "role": "assistant",
            "content": masked_answer
        })
    return state
