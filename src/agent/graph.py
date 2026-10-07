"""Estado, nodos y grafo ReAct del agente.

    START → agent ──(¿tool_calls?)── sí → tools ─┐
              ↑                                   │
              └───────────────────────────────────┘
                       no → END
"""

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage, trim_messages
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .config import Settings
from .tools import TOOLS

SYSTEM_PROMPT = """\
Sos un asistente de consultas sobre clientes y pedidos de una tienda.

Reglas:
- Usá las herramientas para obtener los datos. Nunca inventes IDs, montos, fechas ni estados.
- Si una herramienta devuelve un campo "error", leé la "sugerencia" y reintentá UNA vez con \
otra herramienta o con parámetros corregidos. Si sigue fallando, explicale al usuario qué pasó \
y qué dato necesitás.
- Si una búsqueda devuelve varias coincidencias ("ambiguo": true), no elijas una: pedile al \
usuario que aclare cuál, listando las opciones con nombre e ID.
- Usá el historial de la conversación para resolver referencias como "¿y el último?" o \
"ese cliente", y reutilizá los IDs que ya conocés en vez de volver a buscarlos.
- Si la pregunta no indica de qué cliente o pedido se trata y el historial tampoco, pedí el dato.
- Respondé en español, de forma breve, con los montos en formato $14.500.
"""


class AgentState(MessagesState):
    """Estado del agente: hereda `messages` con el reducer `add_messages`."""


def build_llm(settings: Settings) -> Runnable[LanguageModelInput, BaseMessage]:
    """Crea el modelo de Anthropic vinculado a las herramientas."""
    kwargs: dict[str, Any] = {
        "model": settings.anthropic_model,
        "api_key": settings.anthropic_api_key,
        "max_tokens": 4096,
    }
    if settings.temperature is not None:
        kwargs["temperature"] = settings.temperature
    return ChatAnthropic(**kwargs).bind_tools(TOOLS)


def build_graph(
    checkpointer: BaseCheckpointSaver[Any],
    llm_with_tools: Runnable[LanguageModelInput, BaseMessage],
    max_context_messages: int = 20,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Arma y compila el grafo ReAct con persistencia."""

    async def call_model(state: AgentState) -> dict[str, list[AnyMessage]]:
        # El recorte solo afecta lo que ve el LLM: el checkpoint guarda el historial completo.
        # start_on="human" evita dejar un ToolMessage huérfano al inicio de la ventana.
        window: list[BaseMessage] = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=len,
            max_tokens=max_context_messages,
            start_on="human",
            include_system=False,
        )
        if not window:
            # El turno actual solo no entra en la ventana: se manda entero desde su HumanMessage.
            history = state["messages"]
            last_human = max(i for i, m in enumerate(history) if m.type == "human")
            window = list(history[last_human:])
        response = await llm_with_tools.ainvoke([SystemMessage(SYSTEM_PROMPT), *window])
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    return workflow.compile(checkpointer=checkpointer)
