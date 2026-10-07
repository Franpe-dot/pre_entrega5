"""CLI del agente: demo de escenarios con trazas y chat interactivo.

    python main.py demo
    python main.py chat --thread-id mi-sesion
"""

import argparse
import asyncio
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph

from src.agent.config import Settings, get_settings
from src.agent.graph import build_graph, build_llm
from src.agent.tracing import Turn, final_answer, logger, run_turn, setup_logging, write_trace

TRACES_DIR = Path(__file__).parent / "traces"

Check = Callable[[list[Turn]], bool]


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    thread: str  # los escenarios con la misma clave comparten thread_id (y por lo tanto memoria)
    prompts: list[str]
    check_label: str
    check: Check


def _tools_called(turn: Turn) -> list[str]:
    return [s["tool"] for s in turn["pasos"] if s["tipo"] == "tool_call"]


def _has_error_result(turn: Turn) -> bool:
    return any(
        s["tipo"] == "tool_result" and isinstance(s["contenido"], dict) and "error" in s["contenido"]
        for s in turn["pasos"]
    )


def _check_multipaso(turns: list[Turn]) -> bool:
    return sum(t["tool_calls"] for t in turns) >= 2


def _check_memoria(turns: list[Turn]) -> bool:
    return "buscar_cliente" not in _tools_called(turns[0]) and bool(final_answer(turns[-1]))


def _check_error(turns: list[Turn]) -> bool:
    first, second = turns
    answer = final_answer(second)
    return (
        _has_error_result(first)
        and bool(final_answer(first))
        and "Laura" in answer
        and "Martín" in answer
    )


def _check_aislamiento(turns: list[Turn]) -> bool:
    return turns[0]["tool_calls"] == 0 and "?" in final_answer(turns[0])


SCENARIOS: list[Scenario] = [
    Scenario(
        name="01_multipaso",
        description=(
            "El usuario da un nombre, no un ID: el agente encadena buscar_cliente → "
            "buscar_pedidos antes de responder (razonamiento multi-paso)."
        ),
        thread="laura",
        prompts=["¿Cuántos pedidos tuvo Laura Gómez y cuál fue el total?"],
        check_label="la herramienta se invoca al menos 2 veces",
        check=_check_multipaso,
    ),
    Scenario(
        name="02_memoria",
        description=(
            "Mismo thread_id que 01_multipaso: el agente resuelve '¿y el último?' con el "
            "contexto persistido, sin volver a buscar al cliente."
        ),
        thread="laura",
        prompts=["¿Y el último?", "¿En qué estado está?"],
        check_label="recuerda al cliente (no vuelve a llamar a buscar_cliente)",
        check=_check_memoria,
    ),
    Scenario(
        name="03_error_y_reintento",
        description=(
            "Ciclo de retorno: una herramienta devuelve error (cliente inexistente) y otra "
            "un resultado ambiguo (dos clientes 'Gómez'); el agente explica o pide aclaración."
        ),
        thread="errores",
        prompts=["Mostrame los pedidos del cliente 999", "¿Qué pedidos tiene Gómez?"],
        check_label="maneja el error de la tool y pide aclarar entre Laura y Martín",
        check=_check_error,
    ),
    Scenario(
        name="04_aislamiento_threads",
        description=(
            "Thread nuevo, misma pregunta que en 02: sin historial el agente no sabe de "
            "qué cliente se habla y lo pregunta. La memoria es por thread_id."
        ),
        thread="aislado",
        prompts=["¿Y el último pedido?"],
        check_label="sin contexto no llama herramientas y pide aclaración",
        check=_check_aislamiento,
    ),
]


async def run_demo(app: CompiledStateGraph[Any, Any, Any, Any], settings: Settings) -> None:
    setup_logging(TRACES_DIR / "run.log")
    # Sufijo por corrida: el checkpoint es persistente y no queremos heredar corridas previas.
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    results: list[tuple[Scenario, bool]] = []

    for scenario in SCENARIOS:
        thread_id = f"demo-{scenario.thread}-{run_id}"
        started = datetime.now().astimezone().isoformat(timespec="seconds")
        logger.info("[%s] thread=%s", scenario.name, thread_id)

        turns: list[Turn] = []
        for prompt in scenario.prompts:
            turns.append(await run_turn(app, prompt, thread_id, settings.recursion_limit))
        logger.info("")

        await write_trace(
            TRACES_DIR / f"{scenario.name}.json",
            {
                "escenario": scenario.name,
                "descripcion": scenario.description,
                "thread_id": thread_id,
                "modelo": settings.anthropic_model,
                "recursion_limit": settings.recursion_limit,
                "inicio": started,
                "turnos": turns,
                "tool_calls_total": sum(t["tool_calls"] for t in turns),
            },
        )
        results.append((scenario, scenario.check(turns)))

    # Los checks son orientativos: el LLM no es determinista.
    logger.info("Resumen de checks")
    for scenario, passed in results:
        logger.info(
            "  [%s] %s — %s", "PASS" if passed else "FAIL", scenario.name, scenario.check_label
        )


async def run_chat(
    app: CompiledStateGraph[Any, Any, Any, Any], settings: Settings, thread_id: str
) -> None:
    setup_logging()
    print(f"Chat sobre thread_id='{thread_id}'. Escribí 'salir' para terminar.\n")
    while True:
        try:
            user_input = (await asyncio.to_thread(input, "> ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"salir", "exit", "quit"}:
            break
        if user_input:
            await run_turn(app, user_input, thread_id, settings.recursion_limit)
            print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agente ReAct con memoria persistente")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="Corre los escenarios y genera las trazas en traces/")
    chat = sub.add_parser("chat", help="Chat interactivo sobre un thread persistente")
    chat.add_argument("--thread-id", default="default", help="ID de la sesión a continuar")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = get_settings()

    async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db) as saver:
        app = build_graph(saver, build_llm(settings), settings.max_context_messages)
        if args.command == "demo":
            await run_demo(app, settings)
        else:
            await run_chat(app, settings, args.thread_id)


if __name__ == "__main__":
    # La consola de Windows no siempre es UTF-8 y la traza usa "→".
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    asyncio.run(main())
