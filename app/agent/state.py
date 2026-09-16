from typing import TypedDict
from typing_extensions import Required, NotRequired

from app.schemas import MovieDTO

from app.agent.schemas import MovieQueryPlan, ResolvedMovieReference


class AgentState(TypedDict, total=False):
    request_id: Required[str]
    user_id: Required[int]
    user_query: Required[str]

    query_plan: NotRequired[MovieQueryPlan]

    resolved_references: NotRequired[
        list[ResolvedMovieReference]
    ]

    candidates: NotRequired[list[MovieDTO]]

    final_response: NotRequired[str]