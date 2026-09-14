import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from app.agent.state import AgentState
from app.agent.tools import (
    fetch_guessed_movie_hybrid,
    fetch_recommended_movies,
    generate_query_embedding,
)
from app.schemas import MovieDTO
from schema import MovieQueryPlan
from app.agent.load_prompts import prompts


logger = logging.getLogger("uvicorn")


async def parser_query_node(state: AgentState, llm: ChatOpenAI):
    structed_output_llm = llm.with_structured_output(MovieQueryPlan)

    messages = [
        SystemMessage(content=prompts["SYSTEM_PARSER_QUERY_PROMPT"]),
        HumanMessage(content=state['user_query'])
    ]

    query_plan: MovieQueryPlan = await structed_output_llm.invoke(messages)

    return {
        'query_plan': query_plan,
    }

async def search_movies(state: AgentState, llm: ChatOpenAI):
