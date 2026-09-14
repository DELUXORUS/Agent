from typing import TypedDict
from typing_extensions import Required, NotRequired

from app.agent.schemas import MovieQueryPlan
from app.schemas import MovieDTO


class AgentState(TypedDict, total=False):
    request_id: Required[str]
    user_id: Required[int]
    user_query: Required[str]

    query_plan: NotRequired[MovieQueryPlan]

    resolved_references: NotRequired[list[MovieDTO]]
    candidates: NotRequired[list[MovieDTO]]

    final_response: NotRequired[str]