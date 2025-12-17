from typing_extensions import TypedDict
from typing import List, Optional, Literal, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class SupportState(TypedDict):
    ticket_id: str 
    messages: Annotated[List[AnyMessage], add_messages]
    intent: Literal["refund","general_inquiry"]
    sentiment: Literal["positive", "neutral", "negative", "angry"]
    refund_amount: Optional[float]
    refund_reason: Optional[str]
    requires_approval: bool