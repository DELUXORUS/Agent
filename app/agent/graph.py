from functools import partial
from app.agent.nodes import (
    general_chat,
    intent_classification_node,
    search_for_guess_node,
    search_for_recommended_node,
    synthesis_response_node,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.agent.router import route_intent
from app.agent.state import AgentState
from app.config import settings


# Инициализируем LLM
# llm = ChatOpenAI(
#     model=settings.LLM_MODEL_NAME,
#     openai_api_key=settings.OPENROUTER_API_KEY,
#     openai_api_base=settings.OPENROUTER_URL,
#     temperature=0.1,
# )

llm_strict = ChatOpenAI(
    model=settings.LLM_MODEL_NAME,
    temperature=settings.LLM_STRICT_TEMPERATURE,
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_URL,
)

# Творческая модель для форматирования финального ответа и general_chat
llm_creative = ChatOpenAI(
    model=settings.LLM_MODEL_NAME,
    temperature=settings.LLM_CREATIVE_TEMPERATURE,
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_URL,
)

# 1. Создаем граф
workflow = StateGraph(AgentState)

# 2. Регистрируем узлы
workflow.add_node(
    "intent_classifier", partial(intent_classification_node, llm=llm_strict)
)
workflow.add_node(
    "search_for_recomended", partial(search_for_recommended_node, llm=llm_strict)
)
workflow.add_node(
    "search_for_guess", partial(search_for_guess_node, llm=llm_strict)
)
workflow.add_node("general_chat", partial(general_chat, llm=llm_creative))
workflow.add_node(
    "synthesis_response", partial(synthesis_response_node, llm=llm_creative)
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