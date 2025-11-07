from enum import Enum
from typing import List, Dict, Any
from datetime import datetime
import json
import logging
from src.config import TOP_K, OPENAI_MODEL, MAX_TOOL_HOPS, TEMPERATURE, MAX_NODE_ITERATIONS
from src.graph.state import State, ToolCall, build_initial_state
from src.graph.registry import load_index_and_chunks, load_tools
from src.rag.retriever import retrieve
from src.guards.policy import apply_guards, mask_pii, check_grounding_required
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


def guard_router(state: State) -> str:
    """Route after guard check: finalize if refused, otherwise continue to main router."""
    if state.get("final_answer") and state.get("final_answer").strip():
        return NodeName.FINALIZE.value
    return router(state)


def router(state: State) -> str:
    """Use an LLM to decide the next node to execute."""
    if state["iteration_count"] > MAX_NODE_ITERATIONS:
        logging.warning(
            f"Iteration count exceeded maximum tool hops: {state['iteration_count']}")
        return NodeName.FINALIZE.value

    query = state["query"]

    # Build state context for the router
    has_chunks = len(state.get("retrieved_chunks", [])) > 0
    has_answer = state.get("final_answer") is not None and state.get(
        "final_answer", "").strip() != ""
    has_tool_calls = len(state.get("tool_calls", [])) > 0

    # Summarize what's been retrieved
    chunk_summary = ""
    if has_chunks:
        chunk_summary = "\n".join([
            f"- {chunk['source']}: {chunk['content'][:150]}..."
            for chunk in state.get("retrieved_chunks", [])[:3]
        ])

    # Build context sections
    chunks_section = ""
    if chunk_summary:
        chunks_section = f"Retrieved chunks summary:\n{chunk_summary}\n"

    answer_section = ""
    if has_answer:
        answer_section = f"Final answer so far: {state.get('final_answer', '')[:200]}...\n"

    tool_calls_section = ""
    if has_tool_calls:
        tool_calls_section = f"Tool calls made: {len(state.get('tool_calls', []))} call(s)\n"

    system_prompt = f"""You are a routing model that decides the next step for handling the user's query.

    **KEY DISTINCTION:**
    - **RAG** = STATIC documents (policies, procedures, facts stored in markdown/PDF files)
    - **TOOLS** = DYNAMIC data (calendar events, calculations, current state that changes)
    
    **CHECK FINALIZE FIRST** (highest priority):
    Route to 'finalize' when the query is sufficiently answered. Consider finalizing if:
    - We have a complete final_answer that addresses the query
    - For informational queries: have chunks + answer → finalize
    - For dynamic queries: have tool results + answer → finalize
    - We're repeating the same actions (check decision_history for loops)
    
    **Current State:**
    - Has retrieved chunks (RAG called): {has_chunks}
    - Has final answer: {has_answer}
    - Has tool calls: {has_tool_calls}
    
    {chunks_section}{answer_section}{tool_calls_section}

    **Decision Rules (check in order):**
    
    1. **'finalize'** → Use when:
       - Query is answered (has final_answer that addresses the query)
       - For static document queries: have chunks + answer → finalize
       - For dynamic tool queries: have tool results + answer → finalize
       - If tool_calls exist AND final_answer exists → ALWAYS finalize (don't call tool again)
       - Detecting loops: same tool called 2+ times with same arguments → finalize
    
    2. **'tool'** → Use when query involves DYNAMIC data or actions:
       - Calendar queries (schedule, list events, "what's on my calendar", "book meeting")
       - Calculations (math, "calculate", "compute", "what is X * Y")
       - Any action that changes state or reads current/dynamic data
       - We already have chunks with formulas/data and need to calculate
       - ONLY if we DON'T have tool_calls yet, OR tool_calls exist but no final_answer yet
       - NOTE: Even if query uses "what" or "list", if it's about calendar/events → tool
    
    3. **'rag'** → Use when query needs STATIC document lookup:
       - Policies, procedures, definitions ("what is", "how much allowance", "explain policy")
       - Historical facts stored in documents
       - Formulas/procedures documented in files
       - ONLY if we DON'T have retrieved_chunks yet

    **Examples:**
    - "What is overtime rate?" → 'rag' (static policy document)
    - "What events are on my calendar?" → 'tool' (dynamic calendar data, not in documents)
    - "Schedule a meeting" → 'tool' (action on calendar)
    - "Overtime pay for 3 hours?" → If no chunks: 'rag' (get formula). If have chunks: 'tool' (calculate)
    - "Calculate 10 * 5" → 'tool' (pure math, no lookup needed)
    - "What's the relocation allowance?" → 'rag' (static document fact)
    
    **Critical:** 
    - Calendar/events queries ALWAYS go to 'tool', never 'rag' (calendar is not in documents)
    - "What is" questions can be either: if about documents → rag, if about calendar/current state → tool

    Only reply with one word: either 'rag' or 'tool' or 'finalize'."""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User query: {query}"}
        ],
        temperature=TEMPERATURE,
    )

    decision = response.choices[0].message.content.strip().lower()
    # Basic sanity check
    if decision not in (NodeName.RAG, NodeName.TOOL, NodeName.FINALIZE):
        logging.warning(f"Invalid decision: {decision}")
        return NodeName.FINALIZE.value

    logging.info(f"Router decision: {decision}")
    return decision


def generate_rag_answer(query: str, retrieved_chunks: List[Dict]) -> str:
    """Generate a response using the retrieved chunks."""
    context = "\n".join(
        [f"Document: {chunk['source']}\n{chunk['content']}" for chunk in retrieved_chunks])
    prompt = f"""Based on the following retrieved documents, answer the question. 
    If the answer cannot be found in the documents, say so.

    Documents:
    {context}

    Question: {query}

    Answer (cite sources using [Source: filename]):"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based ONLY on the provided documents. Always cite your sources."},
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def rag_node(state: State) -> State:
    index, chunks = load_index_and_chunks()
    query = state["query"]
    # retrieve chunks
    retrieved_chunks = retrieve(query, index, chunks, k=TOP_K)

    # Check grounding requirement
    is_grounded, grounding_error = check_grounding_required(
        query, retrieved_chunks
    )
    if not is_grounded and grounding_error:
        logging.warning(f"Grounding check failed: {grounding_error}")
        state["final_answer"] = grounding_error
        state["retrieved_chunks"] = retrieved_chunks
        state["citations"] = []
        return state

    # generate answer
    answer = generate_rag_answer(query, retrieved_chunks)
    state["retrieved_chunks"] = retrieved_chunks
    state["final_answer"] = answer
    state["citations"] = list(set(chunk["source"]
                              for chunk in retrieved_chunks))
    state["tool_calls"].append(
        ToolCall(name=NodeName.RAG.value, arguments=query, result=answer, timestamp=datetime.now()))
    state["iteration_count"] += 1
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
        "Begin!"
    )

    agent_scratchpad = ""
    messages = [
        {"role": "system", "content": system.replace("{tools}", tool_names)},
    ]

    final_answer: str | None = None
    for _ in range(MAX_TOOL_HOPS):
        user_content = (
            f"Question: {query}\nThought:{agent_scratchpad}"
        )
        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=tool_definitions,
            temperature=TEMPERATURE,
        )

        choice = response.choices[0].message
        tool_calls = choice.tool_calls

        if not tool_calls:
            final_answer = (choice.content or "").strip()
            break

        # Execute each tool call, append Observation to scratchpad
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool_fn = tools[tool_name]
            try:
                result = tool_fn(**tool_args)
                state["tool_calls"].append(
                    ToolCall(name=tool_name, arguments=tool_args, result=result, timestamp=datetime.now()))
                logging.info(f"Tool call {tool_name} result: {result}")
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                logging.error(f"Tool call {tool_name} failed: {error_msg}")
                result = error_msg
                state["tool_calls"].append(
                    ToolCall(name=tool_name, arguments=tool_args, result=error_msg, timestamp=datetime.now()))

            # Update scratchpad with Observation and a new Thought
            agent_scratchpad += (
                f"\nAction: {tool_name}\nAction Input: {tool_args}\nObservation: {result}\nThought:"
            )

    if not final_answer:
        final_answer = generate_tool_call_answer(query, state["tool_calls"])

    state["final_answer"] = final_answer
    state["iteration_count"] += 1
    logging.info(f"Tool node final answer: {final_answer}")
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
