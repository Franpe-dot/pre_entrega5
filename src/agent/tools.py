"""Herramientas del agente.

El LLM elige qué herramienta usar únicamente a partir del docstring, así que
cada uno explica qué hace, cuándo usarla, qué devuelve y qué hacer ante un error.
Los errores de negocio se devuelven como dato (`error` + `sugerencia`) en vez de
lanzarse: eso es lo que le permite al agente reintentar o pedir aclaraciones.
"""

from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from . import mock_db

ToolResult = dict[str, Any]


class BuscarClienteInput(BaseModel):
    nombre: str = Field(
        min_length=2,
        description="Nombre, apellido o parte del nombre del cliente. Ej: 'Laura Gómez' o 'Gómez'.",
    )


class BuscarPedidosInput(BaseModel):
    cliente_id: int = Field(gt=0, description="ID numérico del cliente. Ej: 102.")


class DetallePedidoInput(BaseModel):
    pedido_id: int = Field(gt=0, description="ID numérico del pedido. Ej: 5003.")


@tool(args_schema=BuscarClienteInput)
async def buscar_cliente(nombre: str) -> ToolResult:
    """Busca clientes por nombre o apellido y devuelve su ID numérico (cliente_id).

    Usar cuando el usuario se refiera a un cliente por su nombre y todavía no se
    conozca su cliente_id; las demás herramientas necesitan ese ID. La búsqueda
    es parcial y no distingue mayúsculas ni tildes.

    Devuelve {"coincidencias": [{"cliente_id", "nombre"}], "ambiguo": bool}.
    Si "ambiguo" es true hay más de un cliente posible: NO elegir uno, pedirle al
    usuario que aclare cuál listando las opciones. Si devuelve un campo "error"
    (ninguna coincidencia), leer la "sugerencia" y actuar en consecuencia.
    """
    try:
        matches = await mock_db.find_clients_by_name(nombre)
    except Exception as exc:
        return {"error": f"Fallo al consultar clientes: {exc}"}

    if not matches:
        return {
            "error": f"No se encontró ningún cliente con el nombre '{nombre}'",
            "sugerencia": "verificar el nombre con el usuario o pedirle el ID del cliente",
        }
    return {
        "coincidencias": [
            {"cliente_id": c["cliente_id"], "nombre": c["nombre"]} for c in matches
        ],
        "ambiguo": len(matches) > 1,
    }


@tool(args_schema=BuscarPedidosInput)
async def buscar_pedidos(cliente_id: int) -> ToolResult:
    """Obtiene todos los pedidos de un cliente a partir de su ID numérico (cliente_id).

    Usar cuando el usuario pregunte por cantidad de pedidos, montos totales,
    último pedido o historial de compras de un cliente. Si el usuario da un
    nombre en vez de un ID, primero usar buscar_cliente.

    Devuelve la cantidad de pedidos, el total gastado y la lista de pedidos
    (pedido_id, fecha, total) ordenada del más reciente al más antiguo. Para ver
    los items o el estado de un pedido, usar obtener_detalle_pedido con su
    pedido_id. Si devuelve un campo "error", leer la "sugerencia" y actuar en
    consecuencia.
    """
    try:
        if not await mock_db.client_exists(cliente_id):
            return {
                "error": f"Cliente {cliente_id} no existe",
                "sugerencia": "usar buscar_cliente para obtener el ID correcto o pedírselo al usuario",
            }
        orders = await mock_db.get_orders_by_client(cliente_id)
    except Exception as exc:
        return {"error": f"Fallo al consultar pedidos: {exc}"}

    return {
        "cliente_id": cliente_id,
        "cantidad_pedidos": len(orders),
        "total": sum(o["total"] for o in orders),
        "pedidos": [
            {"pedido_id": o["pedido_id"], "fecha": o["fecha"], "total": o["total"]}
            for o in orders
        ],
    }


@tool(args_schema=DetallePedidoInput)
async def obtener_detalle_pedido(pedido_id: int) -> ToolResult:
    """Obtiene el detalle completo de un pedido a partir de su ID numérico (pedido_id).

    Usar cuando el usuario pregunte por los productos (items), el estado
    (pendiente, en camino, entregado, cancelado), la fecha o el monto de un
    pedido puntual. El pedido_id se obtiene de buscar_pedidos; no inventarlo.

    Devuelve {"pedido_id", "cliente_id", "fecha", "total", "estado", "items"}.
    Si devuelve un campo "error", leer la "sugerencia" y actuar en consecuencia.
    """
    try:
        order = await mock_db.get_order_detail(pedido_id)
    except Exception as exc:
        return {"error": f"Fallo al consultar el pedido: {exc}"}

    if order is None:
        return {
            "error": f"Pedido {pedido_id} no existe",
            "sugerencia": "usar buscar_pedidos para ver los IDs de pedido válidos del cliente",
        }
    return order


TOOLS: list[BaseTool] = [buscar_cliente, buscar_pedidos, obtener_detalle_pedido]
