import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from app.agent.state import AgentState
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

    query_plan: MovieQueryPlan = await structed_output_llm.ainvoke(messages)

    return {
        'query_plan': query_plan,
    }


async def general_chat(state: AgentState, llm: ChatOpenAI):
    pass

async def movie_request(state: AgentState):
    pass

async def resolve_references(state: AgentState):
    pass

async def search_movies(state: AgentState):
    pass

async def evaluate_results(state: AgentState, llm: ChatOpenAI):
    pass

async def compose_response(state: AgentState, llm: ChatOpenAI):
    pass