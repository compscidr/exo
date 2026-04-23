"""TCP-ring pipeline transport for tinygrad multi-node sharding.

Each rank listens on `bind_port`, waits for rank (k-1) mod N to connect
(receive direction), then opens an outbound connection to rank (k+1) mod N
(send direction). Messages are 5-byte tag+length framed; payloads carry
either hidden-state tensors (fp16) or sampled tokens.
"""

import contextlib
import socket
import struct
import time
from dataclasses import dataclass
from typing import Any, Self, cast

import numpy as np
from loguru import logger

from exo.shared.types.common import Host, NodeId

TAG_HIDDEN = 0
TAG_TOKEN = 1
TAG_STOP = 2

_HEADER = struct.Struct("<BI")  # tag u8, length u32-le
_HIDDEN_HEADER = struct.Struct("<IHIH")  # position_offset u32, batch u16, seq_len u32, hidden_dim u16
_TOKEN_PAYLOAD = struct.Struct("<iB")  # token_id i32-le, stop u8


def encode_hidden(
    arr: np.ndarray[Any, np.dtype[np.uint16]],
    position_offset: int = 0,
) -> bytes:
    # Wire format: <position_offset:u32><batch:u16><seq_len:u32><hidden_dim:u16>
    # followed by raw bf16 bytes (carried as uint16 since numpy has no bf16).
    # position_offset signals which cache position range the hidden should
    # be written at on the receiving rank — 0 means fresh session.
    assert arr.dtype == np.uint16 and arr.ndim == 3
    batch = cast(int, arr.shape[0])
    seq_len = cast(int, arr.shape[1])
    hidden_dim = cast(int, arr.shape[2])
    assert batch < 2**16 and hidden_dim < 2**16, "shape overflows u16"
    assert 0 <= position_offset < 2**32, "position_offset overflows u32"
    header = _HIDDEN_HEADER.pack(position_offset, batch, seq_len, hidden_dim)
    return header + arr.tobytes()


def decode_hidden(
    buf: bytes,
) -> tuple[int, np.ndarray[Any, np.dtype[np.uint16]]]:
    # Returns (position_offset, writable uint16 ndarray carrying bf16 bits).
    raw = _HIDDEN_HEADER.unpack_from(buf, 0)
    position_offset = cast(int, raw[0])
    batch = cast(int, raw[1])
    seq_len = cast(int, raw[2])
    hidden_dim = cast(int, raw[3])
    expected_bytes = batch * seq_len * hidden_dim * 2  # bf16 = 2 bytes/element
    actual_bytes = len(buf) - _HIDDEN_HEADER.size
    assert actual_bytes == expected_bytes, (
        f"decode_hidden: wire corruption — expected {expected_bytes} data bytes, "
        f"got {actual_bytes}"
    )
    data = np.frombuffer(buf[_HIDDEN_HEADER.size:], dtype=np.uint16)
    return position_offset, data.reshape(batch, seq_len, hidden_dim).copy()


def encode_token(token_id: int, stop: bool) -> bytes:
    return _TOKEN_PAYLOAD.pack(token_id, 1 if stop else 0)


def decode_token(buf: bytes) -> tuple[int, bool]:
    raw = _TOKEN_PAYLOAD.unpack_from(buf, 0)
    token_id = cast(int, raw[0])
    stop_int = cast(int, raw[1])
    return token_id, bool(stop_int)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("peer closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_frame(sock: socket.socket, tag: int, payload: bytes) -> None:
    sock.sendall(_HEADER.pack(tag, len(payload)))
    if payload:
        sock.sendall(payload)


def _recv_frame(sock: socket.socket) -> tuple[int, bytes]:
    header = _recv_exactly(sock, _HEADER.size)
    raw = _HEADER.unpack(header)
    tag = cast(int, raw[0])
    length = cast(int, raw[1])
    payload = _recv_exactly(sock, length) if length else b""
    return tag, payload


@dataclass
class PipelineGroup:
    rank: int
    world_size: int
    recv_sock: socket.socket  # from rank (k-1) mod N
    send_sock: socket.socket  # to rank (k+1) mod N

    @classmethod
    def connect(
        cls,
        *,
        rank: int,
        world_size: int,
        hosts_by_node: dict[NodeId, list[Host]],
        bind_port: int,
        node_rank_mapping: list[NodeId],
        connect_timeout: float = 30.0,
        retry_interval: float = 0.1,
    ) -> Self:
        assert 0 <= rank < world_size
        assert len(node_rank_mapping) == world_size

        # hosts_by_node is produced by get_mlx_ring_hosts_by_node: per node,
        # the list is indexed by rank-in-cycle, and entry[r] is the IP this
        # node uses to reach the node at rank r. Entry[self_rank] is 0.0.0.0
        # (the self placeholder). Entry[neighbour_rank] is a real IP.
        next_rank = (rank + 1) % world_size
        self_node = node_rank_mapping[rank]
        self_hosts = hosts_by_node[self_node]
        if len(self_hosts) <= next_rank:
            raise RuntimeError(
                f"hosts_by_node[{self_node}] has {len(self_hosts)} entries, "
                f"need entry for rank {next_rank}"
            )
        next_host = self_hosts[next_rank]
        if next_host.ip in ("0.0.0.0", "198.51.100.1"):
            raise RuntimeError(
                f"pipeline rank {rank}: invalid next-hop IP {next_host.ip!r} "
                f"at hosts_by_node[{self_node}][{next_rank}] — placement did "
                f"not provision neighbour connectivity"
            )

        listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen.bind(("0.0.0.0", bind_port))
        listen.listen(4)
        logger.info(f"pipeline rank {rank}: listening on :{bind_port}")

        deadline = time.monotonic() + connect_timeout
        send_sock: socket.socket | None = None
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((next_host.ip, next_host.port))
                send_sock = s
                break
            except (ConnectionRefusedError, OSError) as e:
                last_err = e
                time.sleep(retry_interval)
        if send_sock is None:
            listen.close()
            raise RuntimeError(
                f"pipeline rank {rank}: could not connect to rank {next_rank} "
                f"at {next_host.ip}:{next_host.port}: {last_err}"
            )
        logger.info(
            f"pipeline rank {rank}: connected to rank {next_rank} "
            f"at {next_host.ip}:{next_host.port}"
        )

        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"pipeline rank {rank}: connect_timeout exhausted before accept(); "
                    f"upstream peer (rank {(rank - 1) % world_size}) never connected"
                )
            listen.settimeout(remaining)
            try:
                recv_sock, _peer_addr = listen.accept()  # pyright: ignore[reportAny]
            except TimeoutError as exc:
                raise RuntimeError(
                    f"pipeline rank {rank}: timed out waiting for inbound connection "
                    f"from upstream peer (rank {(rank - 1) % world_size})"
                ) from exc
            listen.settimeout(None)  # restore blocking mode for later recv
        except Exception:
            listen.close()
            send_sock.close()
            raise
        else:
            listen.close()
        logger.info(f"pipeline rank {rank}: accepted inbound connection")

        for s in (send_sock, recv_sock):
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        return cls(
            rank=rank,
            world_size=world_size,
            recv_sock=recv_sock,
            send_sock=send_sock,
        )

    def send_hidden(
        self,
        arr: np.ndarray[Any, np.dtype[np.uint16]],
        position_offset: int = 0,
    ) -> None:
        _send_frame(self.send_sock, TAG_HIDDEN, encode_hidden(arr, position_offset))

    def recv_hidden(self) -> tuple[int, np.ndarray[Any, np.dtype[np.uint16]]]:
        tag, payload = _recv_frame(self.recv_sock)
        if tag != TAG_HIDDEN:
            raise RuntimeError(f"expected HIDDEN, got tag={tag}")
        return decode_hidden(payload)

    def send_token(self, token_id: int, stop: bool) -> None:
        _send_frame(self.send_sock, TAG_TOKEN, encode_token(token_id, stop))

    def recv_token(self) -> tuple[int, bool]:
        tag, payload = _recv_frame(self.recv_sock)
        if tag != TAG_TOKEN:
            raise RuntimeError(f"expected TOKEN, got tag={tag}")
        return decode_token(payload)

    def send_stop(self) -> None:
        _send_frame(self.send_sock, TAG_STOP, b"")

    def recv_any(self) -> tuple[int, bytes]:
        return _recv_frame(self.recv_sock)

    def close(self) -> None:
        for s in (self.recv_sock, self.send_sock):
            with contextlib.suppress(OSError):
                s.shutdown(socket.SHUT_RDWR)
            s.close()
