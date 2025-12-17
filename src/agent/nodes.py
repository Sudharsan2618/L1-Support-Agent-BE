"""
This file contains nodes for this project sentinal.
"""

import os
from typing import Literal, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import interrupt, Command
from langchain_core.prompts import ChatPromptTemplate
from agent.state import SupportState
from pydantic import BaseModel, Field

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",
                            temperature=0
                            )

class TriageData(BaseModel):
    intent: Literal["refund", "general_inquiry"] = Field(
        ..., description="The main intent of the user."
    )
    sentiment: Literal["positive", "neutral", "negative", "angry"] = Field(
        ..., description="The emotional tone of the user."
    )
    refund_amount: float = Field(
        0.0, description="The amount of money requested for refund, if any."
    )
    refund_reason: str = Field(
        None, description="The reason stated for the refund request."
    )

def triage_node(state: SupportState) -> Dict[str, Any]:
    """Analyze the convo and extract
    1. Intent
    2. Sentiment
    3. Refund amount
    4. Refund reason"""
    structured_llm = llm.with_structured_output(TriageData)

    try:
        result = structured_llm.invoke(state["messages"])
        if result is None:
            raise ValueError("No result from structured LLM")
            
        return {
            "intent": result.intent,
            "sentiment": result.sentiment,
            "refund_amount": result.refund_amount,
            "refund_reason": result.refund_reason,
        }
    except Exception as e:
        print(f"Error in triage_node: {e}. Defaulting to general_inquiry.")
        return {
            "intent": "general_inquiry",
            "sentiment": "neutral",
            "refund_amount": 0.0,
            "refund_reason": None,
        }

def policy_check_node(state: SupportState) -> Dict[str, Any]:
    """Check if the refund amount is within policy limits"""
    amount = state.get("refund_amount",0.0)
    if amount>=50.0:
        return {"requires_approval": True}
    else:
        return {"requires_approval": False}

def human_approval_node(state: SupportState) -> Dict[str, Any]:
    """This node pause execution until human approval"""
    print(f"Suspending for human approval")

    user_action = interrupt({
        "type": "appoval_required",
        "amount": state.get("refund_amount", 0.0),
        "reason": state.get("refund_reason") or "High value refund"
    })
    
    print(f"User action received: {user_action}")

    if isinstance(user_action, dict) and "status" in user_action:
        decision = user_action["status"]
    else:
        decision = str(user_action)
    print(f"Decision received: {decision}")
    
    if decision.lower().startswith("approve"):
        return {"requires_approval": False, "messages": [SystemMessage(content="Human approved the refund request")]}
    else:
        return {"intent": "general_inquiry", "messages": [SystemMessage(content="Human denied the refund request")]}

def process_refund_node(state: SupportState) -> Dict[str, Any]:
    """Process the refund request"""
    amount = state['refund_amount']
    print(f"Processing refund of {amount}")
    return {"messages": [SystemMessage(content=f"Processing refund of {amount}")]}

def general_response_node(state: SupportState) -> Dict[str, Any]:
    """
    Standard chat response for non-refund queries or rejected refunds.
    """
    # Simple chat response
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
