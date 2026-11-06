"""Minimal in-memory Redis stand-in for unit tests (Streams + GET/SET only)."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import redis


class MemoryRedis:
    """Enough of redis-py surface for JobQueue + Autoscaler tests without a daemon."""

    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)
        self.groups: dict[str, dict[str, Any]] = defaultdict(dict)
        self.pending: dict[str, dict[str, list[tuple[str, dict[str, str]]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.kv: dict[str, str] = {}
        self._seq = 0

    def xgroup_create(self, name: str, groupname: str, id: str = "0", mkstream: bool = False) -> bool:
        if mkstream and name not in self.streams:
            self.streams[name] = []
        if groupname in self.groups[name]:
            raise redis.ResponseError("BUSYGROUP Consumer Group name already exists")
        self.groups[name][groupname] = {"cursor_index": 0}
        return True

    def xadd(self, name: str, fields: dict[str, str]) -> str:
        self._seq += 1
        msg_id = f"{int(time.time() * 1000)}-{self._seq}"
        self.streams[name].append((msg_id, {str(k): str(v) for k, v in fields.items()}))
        return msg_id

    def xreadgroup(
        self,
        groupname: str,
        consumername: str,
        streams: dict[str, str],
        count: int = 1,
        block: int | None = None,  # noqa: ARG002
    ) -> list:
        out: list = []
        for stream_name, _ in streams.items():
            if groupname not in self.groups[stream_name]:
                raise redis.ResponseError("NOGROUP")
            delivered: list[tuple[str, dict[str, str]]] = []
            group = self.groups[stream_name][groupname]
            idx = int(group.get("cursor_index", 0))
            msgs = self.streams[stream_name]
            taken = 0
            while idx < len(msgs) and taken < count:
                msg_id, fields = msgs[idx]
                delivered.append((msg_id, fields))
                self.pending[stream_name][consumername].append((msg_id, fields))
                idx += 1
                taken += 1
            group["cursor_index"] = idx
            if delivered:
                out.append((stream_name, delivered))
        return out

    def xack(self, name: str, groupname: str, *ids: str) -> int:  # noqa: ARG002
        id_set = set(ids)
        acked = 0
        for consumer, plist in list(self.pending[name].items()):
            keep = []
            for msg_id, fields in plist:
                if msg_id in id_set:
                    acked += 1
                else:
                    keep.append((msg_id, fields))
            self.pending[name][consumer] = keep
        return acked

    def xpending(self, name: str, groupname: str) -> dict[str, Any]:  # noqa: ARG002
        total = sum(len(v) for v in self.pending[name].values())
        return {"pending": total, "min": None, "max": None, "consumers": []}

    def xinfo_groups(self, name: str) -> list[dict[str, Any]]:
        result = []
        length = len(self.streams.get(name, []))
        for gname, g in self.groups[name].items():
            cursor = int(g.get("cursor_index", 0))
            pending = sum(len(v) for v in self.pending[name].values())
            lag = max(0, length - cursor)
            result.append({"name": gname, "lag": lag, "pending": pending})
        return result

    def xlen(self, name: str) -> int:
        return len(self.streams.get(name, []))

    def set(self, key: str, value: str, ex: int | None = None) -> bool:  # noqa: ARG002
        self.kv[key] = value
        return True

    def get(self, key: str) -> str | None:
        return self.kv.get(key)
