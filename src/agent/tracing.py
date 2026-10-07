"""Ejecución de un turno del agente capturando la traza ReAct paso a paso."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger("agent.trace")

Step = dict[str, Any]
Turn = dict[str, Any]

_LOG_PREVIEW_CHARS = 300


def setup_logging(log_file: Path | None = None) -> None:
    """Configura el logger de trazas: consola y, opcionalmente, archivo."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def _text_of(content: str | list[Any]) -> str:
    """Extrae el texto de un mensaje, ignorando bloques que no sean de texto (thinking, tool_use)."""
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts).strip()


def _parse_tool_content(content: str | list[Any]) -> Any:
    """Devuelve el resultado de la tool como dict si es JSON; si no, como texto."""
    text = _text_of(content)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _format_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())


def _preview(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= _LOG_PREVIEW_CHARS else text[:_LOG_PREVIEW_CHARS] + "…"


def _steps_from_update(node: str, messages: list[Any]) -> list[Step]:
    """Convierte la actualización de un nodo en pasos de la traza."""
    steps: list[Step] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            reasoning = _text_of(message.content)
            for call in message.tool_calls:
                step: Step = {
                    "nodo": node,
                    "tipo": "tool_call",
                    "tool": call["name"],
                    "args": call["args"],
                    "id": call["id"],
                }
                if reasoning:
                    step["razonamiento"] = reasoning
                steps.append(step)
        elif isinstance(message, AIMessage):
            steps.append(
                {"nodo": node, "tipo": "final_answer", "contenido": _text_of(message.content)}
            )
        elif isinstance(message, ToolMessage):
            steps.append(
                {
                    "nodo": node,
                    "tipo": "tool_result",
                    "tool": message.name,
                    "tool_call_id": message.tool_call_id,
                    "contenido": _parse_tool_content(message.content),
                }
            )
    return steps


def _log_step(step: Step) -> None:
    match step["tipo"]:
        case "tool_call":
            if "razonamiento" in step:
                logger.info("→ agent razona: %s", _preview(step["razonamiento"]))
            logger.info("→ agent decide: %s(%s)", step["tool"], _format_args(step["args"]))
        case "tool_result":
            logger.info("→ tool devuelve: %s", _preview(step["contenido"]))
        case "final_answer":
            logger.info('Respuesta: "%s"', step["contenido"])
        case "error":
            logger.error("✗ %s", step["contenido"])


async def run_turn(
    app: CompiledStateGraph[Any, Any, Any, Any],
    user_input: str,
    thread_id: str,
    recursion_limit: int,
) -> Turn:
    """Ejecuta un turno sobre `thread_id` y devuelve su traza.

    El `recursion_limit` es el freno del ciclo agent↔tools: si se alcanza, el
    turno termina con un paso de tipo `error` en vez de propagar la excepción.
    """
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    steps: list[Step] = []
    logger.info('Usuario: "%s"', user_input)

    try:
        async for update in app.astream(
            {"messages": [HumanMessage(user_input)]}, config, stream_mode="updates"
        ):
            for node, node_update in update.items():
                if not isinstance(node_update, dict):
                    continue
                for step in _steps_from_update(node, node_update.get("messages", [])):
                    step = {"paso": len(steps) + 1, **step}
                    steps.append(step)
                    _log_step(step)
    except GraphRecursionError:
        step = {
            "paso": len(steps) + 1,
            "nodo": "graph",
            "tipo": "error",
            "contenido": (
                f"Se alcanzó el límite de {recursion_limit} pasos sin llegar a una "
                "respuesta. Reformulá la consulta o dame un dato más preciso."
            ),
        }
        steps.append(step)
        _log_step(step)

    return {
        "usuario": user_input,
        "pasos": steps,
        "tool_calls": sum(1 for s in steps if s["tipo"] == "tool_call"),
    }


def final_answer(turn: Turn) -> str:
    """Texto de la última respuesta (o error) del turno."""
    for step in reversed(turn["pasos"]):
        if step["tipo"] in ("final_answer", "error"):
            return str(step["contenido"])
    return ""


async def write_trace(path: Path, trace: dict[str, Any]) -> None:
    """Guarda la traza de un escenario como JSON sin bloquear el event loop."""
    payload = json.dumps(trace, ensure_ascii=False, indent=2) + "\n"
    await asyncio.to_thread(_write_text, path, payload)


def _write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
