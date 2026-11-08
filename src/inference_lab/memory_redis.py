"""Minimal in-memory Redis stand-in for unit tests (Streams + GET/SET only)."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import redis


class MemoryRedis:
    """Enough of redis-py for JobQueue tests without a daemon.

    Pending entries remember when they were delivered so XAUTOCLAIM idle time
    can be tested. This is not a Redis compatibility suite.
    """

    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)
        self.groups: dict[str, dict[str, Any]] = defaultdict(dict)
        # stream -> group -> consumer -> (id, fields, delivered_ms)
        self.pending: dict[str, dict[str, dict[str, list[tuple[str, dict[str, str], float]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
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
        now_ms = time.time() * 1000
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
                self.pending[stream_name][groupname][consumername].append((msg_id, fields, now_ms))
                idx += 1
                taken += 1
            group["cursor_index"] = idx
            if delivered:
                out.append((stream_name, delivered))
        return out

    def xack(self, name: str, groupname: str, *ids: str) -> int:
        id_set = set(ids)
        acked = 0
        group = self.pending[name][groupname]
        for consumer, plist in list(group.items()):
            keep = []
            for msg_id, fields, delivered_ms in plist:
                if msg_id in id_set:
                    acked += 1
                else:
                    keep.append((msg_id, fields, delivered_ms))
            group[consumer] = keep
        return acked

    def xpending(self, name: str, groupname: str) -> dict[str, Any]:
        total = sum(len(v) for v in self.pending[name][groupname].values())
        return {"pending": total, "min": None, "max": None, "consumers": []}

    def xinfo_groups(self, name: str) -> list[dict[str, Any]]:
        result = []
        length = len(self.streams.get(name, []))
        for gname, g in self.groups[name].items():
            cursor = int(g.get("cursor_index", 0))
            pending = sum(len(v) for v in self.pending[name][gname].values())
            lag = max(0, length - cursor)
            result.append({"name": gname, "lag": lag, "pending": pending})
        return result

    def xautoclaim(
        self,
        name: str,
        groupname: str,
        consumername: str,
        min_idle_time: int,
        start_id: str = "0-0",  # noqa: ARG002
        count: int | None = None,
    ) -> list:
        if groupname not in self.groups[name]:
            raise redis.ResponseError("NOGROUP")
        now_ms = time.time() * 1000
        group = self.pending[name][groupname]
        candidates: list[tuple[str, dict[str, str]]] = []
        for plist in group.values():
            for msg_id, fields, delivered_ms in plist:
                if now_ms - delivered_ms >= float(min_idle_time):
                    candidates.append((msg_id, fields))
        candidates.sort(key=lambda item: item[0])
        if count is not None:
            candidates = candidates[: int(count)]
        chosen = {msg_id for msg_id, _ in candidates}
        for consumer, plist in list(group.items()):
            group[consumer] = [row for row in plist if row[0] not in chosen]
        dest = group[consumername]
        for msg_id, fields in candidates:
            dest.append((msg_id, fields, now_ms))
        return ["0-0", candidates, []]

    def xlen(self, name: str) -> int:
        return len(self.streams.get(name, []))

    def set(self, key: str, value: str, ex: int | None = None, nx: bool = False, **_kwargs: Any) -> bool | None:  # noqa: ARG002
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

    def get(self, key: str) -> str | None:
        return self.kv.get(key)

    def incr(self, key: str) -> int:
        current = int(self.kv.get(key, "0")) + 1
        self.kv[key] = str(current)
        return current
