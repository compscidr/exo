import socket
import threading
from typing import cast

import numpy as np
import pytest

from exo.shared.types.common import Host, NodeId
from exo.worker.engines.tinygrad.pipeline_group import (
    PipelineGroup,
    decode_hidden,
    decode_token,
    encode_hidden,
    encode_token,
)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = cast(int, s.getsockname()[1])
    s.close()
    return port


def test_encode_decode_token_roundtrip() -> None:
    buf = encode_token(42, stop=True)
    token, stop = decode_token(buf)
    assert token == 42 and stop is True


def test_encode_decode_hidden_roundtrip() -> None:
    arr = np.arange(12, dtype=np.uint16).reshape(1, 3, 4)
    buf = encode_hidden(arr)
    out = decode_hidden(buf)
    assert out.shape == arr.shape  # pyright: ignore[reportAny]
    assert np.array_equal(out, arr)


def test_two_rank_ring_localhost_token_roundtrip() -> None:
    node0 = NodeId("node0")
    node1 = NodeId("node1")

    port_a = free_port()
    port_b = free_port()

    # hosts_by_node matches get_mlx_ring_hosts_by_node: per-node list indexed
    # by rank-in-cycle. Entry[self_rank] = 0.0.0.0 placeholder. Entry[other_rank]
    # = IP this node uses to reach that rank.
    hosts: dict[NodeId, list[Host]] = {
        node0: [Host(ip="0.0.0.0", port=port_a), Host(ip="127.0.0.1", port=port_b)],
        node1: [Host(ip="127.0.0.1", port=port_a), Host(ip="0.0.0.0", port=port_b)],
    }

    groups: list[PipelineGroup | None] = [None, None]

    def run_rank(rank: int) -> None:
        g = PipelineGroup.connect(
            rank=rank,
            world_size=2,
            hosts_by_node=hosts,
            bind_port=port_a if rank == 0 else port_b,
            node_rank_mapping=[node0, node1],
        )
        groups[rank] = g

    t0 = threading.Thread(target=run_rank, args=(0,))
    t1 = threading.Thread(target=run_rank, args=(1,))
    t0.start()
    t1.start()
    t0.join(timeout=10)
    t1.join(timeout=10)

    assert groups[0] is not None, "rank 0 failed to connect"
    assert groups[1] is not None, "rank 1 failed to connect"
    assert groups[0].rank == 0 and groups[0].world_size == 2
    assert groups[1].rank == 1 and groups[1].world_size == 2

    # In the ring, rank 1's "next" is rank 0 (wraps around).
    # So rank 1 sending a token goes into rank 0's recv_sock.
    groups[1].send_token(99, stop=False)
    token, stop = groups[0].recv_token()
    assert token == 99 and not stop

    # Hidden state: rank 0 -> rank 1
    arr = np.arange(24, dtype=np.uint16).reshape(1, 6, 4)
    groups[0].send_hidden(arr)
    recv = groups[1].recv_hidden()
    assert recv.shape == arr.shape  # pyright: ignore[reportAny]
    assert np.array_equal(recv, arr)

    groups[0].close()
    groups[1].close()


def test_accept_timeout_raises_runtime_error() -> None:
    """Rank 0 binds and waits for an upstream peer that never arrives.

    With a very short connect_timeout the call must raise RuntimeError (not hang).
    Rank 0's next-rank (rank 1) is reachable for the outbound connect leg
    because we pre-bind rank 1's port and accept that single connection in
    a helper thread, so the outbound-connect loop always succeeds quickly.
    Only the *inbound* accept (upstream from rank 1 → rank 0) is left
    unserviced, so the timeout fires on accept().
    """
    node0 = NodeId("node0")
    node1 = NodeId("node1")

    port_a = free_port()
    port_b = free_port()

    # Pre-bind rank 1's port so rank 0's outbound connect to it succeeds
    # immediately, leaving only the inbound accept() to time out.
    rank1_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    rank1_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rank1_listener.bind(("127.0.0.1", port_b))
    rank1_listener.listen(4)

    accepted_conns: list[socket.socket] = []

    def accept_one() -> None:
        conn, _ = rank1_listener.accept()  # pyright: ignore[reportAny]
        accepted_conns.append(conn)

    acceptor = threading.Thread(target=accept_one, daemon=True)
    acceptor.start()

    hosts: dict[NodeId, list[Host]] = {
        node0: [Host(ip="0.0.0.0", port=port_a), Host(ip="127.0.0.1", port=port_b)],
        node1: [Host(ip="127.0.0.1", port=port_a), Host(ip="0.0.0.0", port=port_b)],
    }

    try:
        with pytest.raises(RuntimeError, match="timed out waiting for inbound connection"):
            PipelineGroup.connect(
                rank=0,
                world_size=2,
                hosts_by_node=hosts,
                bind_port=port_a,
                node_rank_mapping=[node0, node1],
                connect_timeout=0.5,  # very short; inbound peer never comes
            )
    finally:
        rank1_listener.close()
        acceptor.join(timeout=2)
        for c in accepted_conns:
            c.close()
