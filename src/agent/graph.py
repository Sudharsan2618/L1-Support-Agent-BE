"""
Langgraph agent for Invoice Processing System
"""

from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import SupportState
from agent.nodes import (
    triage_node,
    policy_check_node,
    human_approval_node,
    process_refund_node,
    general_response_node
)

def route_after_triage(state: SupportState) -> str:
    if state["intent"] == "refund":
        return "policy_check"
    return "general_response"

def route_after_policy_check(state: SupportState) -> str:
    if state["requires_approval"]:
        return "human_approval"
    return "process_refund"

def route_after_approval(state: SupportState) -> str:
    if not state["requires_approval"]:
        return "process_refund"
    return "general_response"

builder =   StateGraph(SupportState)

builder.add_node("triage", triage_node)
builder.add_node("policy_check", policy_check_node)
builder.add_node("human_approval", human_approval_node)
builder.add_node("process_refund", process_refund_node)
builder.add_node("general_response", general_response_node)

builder.add_edge(START, "triage")
builder.add_conditional_edges("triage", route_after_triage)
builder.add_conditional_edges("policy_check", route_after_policy_check)
builder.add_conditional_edges("human_approval", route_after_approval)
builder.add_edge("process_refund", END)
builder.add_edge("general_response", END)
# checkpoint = MemorySaver()
graph = builder.compile()