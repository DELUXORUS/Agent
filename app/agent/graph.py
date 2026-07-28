# app/agent/graph.py
from functools import partial
from app.agent.nodes import (
    general_chat,
    intent_classification_node,
    search_for_guess_node,
    search_for_recomended_node,
    synthesis_response_node,
)
from app.agent.router import route_intent
from app.agent.state import AgentState
from app.config import settings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Инициализируем LLM
llm = ChatOpenAI(
    model="google/gemma-4-26b-a4b-it:free",
    openai_api_key=settings.OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1,
)

# 1. Создаем граф
workflow = StateGraph(AgentState)

# 2. Регистрируем узлы
workflow.add_node(
    "intent_classifier", partial(intent_classification_node, llm=llm)
)
workflow.add_node(
    "search_for_recomended", partial(search_for_recomended_node, llm=llm)
)
workflow.add_node(
    "search_for_guess", partial(search_for_guess_node, llm=llm)
)
workflow.add_node("general_chat", partial(general_chat, llm=llm))
workflow.add_node(
    "synthesis_response", partial(synthesis_response_node, llm=llm)
)

# 3. Точка входа
workflow.set_entry_point("intent_classifier")

# 4. Развилка (Conditional Edge)
workflow.add_conditional_edges(
    source="intent_classifier",
    path=route_intent,
    path_map={
        "parse_guess_filter": "search_for_guess",
        "parse_recommend_filter": "search_for_recomended",
        "general_chat": "general_chat",
    },
)

# 5. Прямые связи от узлов RAG-поиска к узлу синтеза
workflow.add_edge("search_for_guess", "synthesis_response")
workflow.add_edge("search_for_recomended", "synthesis_response")

# 6. ВЫХОДНЫЕ СВЯЗИ В END
workflow.add_edge("synthesis_response", END)
workflow.add_edge("general_chat", END)

# Компилируем граф
graph = workflow.compile()