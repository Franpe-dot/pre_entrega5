"""Base de datos simulada de clientes y pedidos.

Los datos viven en memoria; cada consulta es `async` y agrega una pequeña
latencia para imitar el I/O de una base real.
"""

import asyncio
import unicodedata
from typing import Any

Record = dict[str, Any]

_LATENCY_SECONDS = 0.1

_CLIENTS: dict[int, Record] = {
    101: {"cliente_id": 101, "nombre": "Carlos Pérez", "email": "carlos.perez@example.com"},
    102: {"cliente_id": 102, "nombre": "Laura Gómez", "email": "laura.gomez@example.com"},
    103: {"cliente_id": 103, "nombre": "Sofía Ramírez", "email": "sofia.ramirez@example.com"},
    104: {"cliente_id": 104, "nombre": "Martín Gómez", "email": "martin.gomez@example.com"},
    105: {"cliente_id": 105, "nombre": "Ana Torres", "email": "ana.torres@example.com"},
    106: {"cliente_id": 106, "nombre": "Diego Fernández", "email": "diego.fernandez@example.com"},
}

_ORDERS: dict[int, Record] = {
    4001: {"pedido_id": 4001, "cliente_id": 101, "fecha": "2026-06-03", "total": 2100, "estado": "entregado",
           "items": [{"producto": "Mouse inalámbrico", "cantidad": 1, "precio_unitario": 2100}]},
    4002: {"pedido_id": 4002, "cliente_id": 101, "fecha": "2026-08-11", "total": 9800, "estado": "entregado",
           "items": [{"producto": "Monitor 24\"", "cantidad": 1, "precio_unitario": 9800}]},
    4003: {"pedido_id": 4003, "cliente_id": 101, "fecha": "2026-09-30", "total": 1500, "estado": "en camino",
           "items": [{"producto": "Cable HDMI", "cantidad": 3, "precio_unitario": 500}]},
    5001: {"pedido_id": 5001, "cliente_id": 102, "fecha": "2026-07-15", "total": 3800, "estado": "entregado",
           "items": [{"producto": "Teclado mecánico", "cantidad": 1, "precio_unitario": 3800}]},
    5002: {"pedido_id": 5002, "cliente_id": 102, "fecha": "2026-08-20", "total": 4500, "estado": "entregado",
           "items": [{"producto": "Auriculares Bluetooth", "cantidad": 1, "precio_unitario": 3000},
                     {"producto": "Funda para notebook", "cantidad": 1, "precio_unitario": 1500}]},
    5003: {"pedido_id": 5003, "cliente_id": 102, "fecha": "2026-09-28", "total": 6200, "estado": "en camino",
           "items": [{"producto": "Webcam Full HD", "cantidad": 1, "precio_unitario": 4200},
                     {"producto": "Hub USB-C", "cantidad": 1, "precio_unitario": 2000}]},
    6001: {"pedido_id": 6001, "cliente_id": 103, "fecha": "2026-05-22", "total": 12000, "estado": "entregado",
           "items": [{"producto": "Silla ergonómica", "cantidad": 1, "precio_unitario": 12000}]},
    6002: {"pedido_id": 6002, "cliente_id": 103, "fecha": "2026-09-05", "total": 2600, "estado": "cancelado",
           "items": [{"producto": "Lámpara de escritorio", "cantidad": 2, "precio_unitario": 1300}]},
    6003: {"pedido_id": 6003, "cliente_id": 103, "fecha": "2026-10-01", "total": 800, "estado": "pendiente",
           "items": [{"producto": "Pad para mouse", "cantidad": 1, "precio_unitario": 800}]},
    7001: {"pedido_id": 7001, "cliente_id": 104, "fecha": "2026-07-02", "total": 5400, "estado": "entregado",
           "items": [{"producto": "Disco SSD 1TB", "cantidad": 1, "precio_unitario": 5400}]},
    7002: {"pedido_id": 7002, "cliente_id": 104, "fecha": "2026-08-30", "total": 1900, "estado": "entregado",
           "items": [{"producto": "Memoria USB 128GB", "cantidad": 2, "precio_unitario": 950}]},
    7003: {"pedido_id": 7003, "cliente_id": 104, "fecha": "2026-09-19", "total": 7300, "estado": "entregado",
           "items": [{"producto": "Router WiFi 6", "cantidad": 1, "precio_unitario": 7300}]},
    7004: {"pedido_id": 7004, "cliente_id": 104, "fecha": "2026-10-04", "total": 3100, "estado": "pendiente",
           "items": [{"producto": "Parlante portátil", "cantidad": 1, "precio_unitario": 3100}]},
    8001: {"pedido_id": 8001, "cliente_id": 105, "fecha": "2026-06-18", "total": 4700, "estado": "entregado",
           "items": [{"producto": "Tablet gráfica", "cantidad": 1, "precio_unitario": 4700}]},
    8002: {"pedido_id": 8002, "cliente_id": 105, "fecha": "2026-08-07", "total": 2250, "estado": "entregado",
           "items": [{"producto": "Lápiz óptico", "cantidad": 1, "precio_unitario": 2250}]},
    8003: {"pedido_id": 8003, "cliente_id": 105, "fecha": "2026-09-12", "total": 6050, "estado": "entregado",
           "items": [{"producto": "Impresora multifunción", "cantidad": 1, "precio_unitario": 6050}]},
    # El cliente 106 existe pero todavía no tiene pedidos.
}


def _normalize(text: str) -> str:
    """Pasa a minúsculas y quita tildes para comparar nombres sin falsos negativos."""
    decomposed = unicodedata.normalize("NFD", text.casefold().strip())
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


async def client_exists(client_id: int) -> bool:
    await asyncio.sleep(_LATENCY_SECONDS)
    return client_id in _CLIENTS


async def find_clients_by_name(name: str) -> list[Record]:
    """Devuelve los clientes cuyo nombre contiene `name` (sin distinguir tildes ni mayúsculas)."""
    await asyncio.sleep(_LATENCY_SECONDS)
    needle = _normalize(name)
    return [dict(c) for c in _CLIENTS.values() if needle in _normalize(c["nombre"])]


async def get_orders_by_client(client_id: int) -> list[Record]:
    """Devuelve los pedidos del cliente ordenados del más reciente al más antiguo."""
    await asyncio.sleep(_LATENCY_SECONDS)
    orders = [dict(o) for o in _ORDERS.values() if o["cliente_id"] == client_id]
    return sorted(orders, key=lambda o: o["fecha"], reverse=True)


async def get_order_detail(order_id: int) -> Record | None:
    await asyncio.sleep(_LATENCY_SECONDS)
    order = _ORDERS.get(order_id)
    return dict(order) if order is not None else None
