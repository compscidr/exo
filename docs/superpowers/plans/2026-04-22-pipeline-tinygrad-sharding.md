# Pipeline-Parallel Tinygrad Sharding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable `Sharding.Pipeline` placement across multiple Nvidia/Linux nodes using tinygrad, so a model too big for any one GPU's VRAM can run split across the cluster (e.g. 20 GB model across a 16 GB + 10 GB pair).

**Architecture:** Add a small TCP-socket-based pipeline group to the tinygrad engine. Each rank listens on `ephemeral_port` and opens one outbound connection to rank `(k+1) % world_size` — forming a ring. Forward pass flows rank 0 → 1 → … → n-1; sampled tokens flow rank n-1 → 0. Rank 0 drives prefill/decode and yields `GenerationResponse` chunks as usual; intermediate ranks enter a loop that receives hidden states, runs their layer slice, and forwards. Middle and last ranks load only the subset of layers they own (weight loader already supports `start_layer`/`end_layer`). Tensor sharding remains single-node; multi-node placement only permitted for `Sharding.Pipeline`.

**Tech Stack:** Python 3.13, tinygrad, anyio (already used in runner IPC), stdlib `socket` + `struct`, pydantic, pytest-asyncio. No new dependencies.

---

## File Structure

**New files:**
- `src/exo/worker/engines/tinygrad/pipeline_group.py` — TCP transport + framing
- `src/exo/worker/engines/tinygrad/tests/test_pipeline_group.py` — unit tests (loopback)
- `src/exo/worker/engines/tinygrad/tests/test_pipeline_integration.py` — two-process end-to-end

**Modified files:**
- `src/exo/master/placement.py` — populate `hosts_by_node`/`ephemeral_port` for multi-node tinygrad; reject multi-node Tensor
- `src/exo/worker/runner/runner_supervisor.py` or `bootstrap.py` — none expected; runner state machine change is local to `tinygrad_runner.py`
- `src/exo/worker/runner/llm_inference/tinygrad_runner.py` — handle `ConnectToGroup`, pass `group` through to generate / warmup / cleanup
- `src/exo/worker/engines/tinygrad/utils_tinygrad.py` — `initialize_tinygrad` now accepts `group`; `load_tinygrad_items` unchanged (shard already layer-bounded)
- `src/exo/worker/engines/tinygrad/generator/generate.py` — thread `group` through `tinygrad_generate` and `warmup_inference`; branch on `group.rank` for pipeline loop
- `src/exo/worker/engines/tinygrad/forward.py` — respect `start_layer`/`end_layer` (skip embed if `start_layer>0`, skip final norm + lm_head if `end_layer<n_layers`); accept optional input `hidden_state` tensor instead of `input_ids` for non-rank-0
- `src/exo/worker/engines/tinygrad/weight_loader.py` — gate `embed_tokens` / `lm_head` / `final_norm` loading by layer bounds (load only when this rank owns the boundary)
- `src/exo/shared/types/worker/runners.py` — add `RunnerConnecting` / `RunnerConnected` states if not already present (check first; they already exist for MLX runner)

---

## Task 1: Gate multi-node placement to Pipeline sharding only

**Files:**
- Modify: `src/exo/master/placement.py:196-200`
- Test: `src/exo/master/tests/test_placement.py` (extend)

- [ ] **Step 1: Write failing test** in `src/exo/master/tests/test_placement.py`

Add at end of file:

```python
def test_tinygrad_multinode_requires_pipeline():
    from exo.shared.types.worker.shards import Sharding
    # two-node cluster
    placements = make_test_placements(
        n_nodes=2,
        instance_meta=InstanceMeta.Tinygrad,
        sharding=Sharding.Tensor,
    )
    with pytest.raises(ValueError, match="multi-node Tinygrad requires Sharding.Pipeline"):
        get_instance_placements(...)  # fill in per helpers in this file

def test_tinygrad_multinode_pipeline_populates_hosts():
    placements = get_instance_placements(
        # Pipeline sharding across 2 nodes
        ...
    )
    (inst,) = placements.values()
    assert isinstance(inst, TinygradInstance)
    assert inst.hosts_by_node is not None
    assert len(inst.hosts_by_node) == 2
    assert inst.ephemeral_port is not None
```

If `make_test_placements` helper doesn't exist, use the existing test patterns in the file to build the input (see other tests in `test_placement.py`).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/exo/master/tests/test_placement.py -k tinygrad_multinode -v`
Expected: FAIL (two tests fail; first fails because no error is raised, second fails because `hosts_by_node` is None)

- [ ] **Step 3: Implement in `src/exo/master/placement.py:196-200`**

Replace:

```python
        case InstanceMeta.Tinygrad:
            target_instances[instance_id] = TinygradInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
            )
```

with:

```python
        case InstanceMeta.Tinygrad:
            n_nodes = len(selected_cycle.node_ids)
            if n_nodes == 1:
                target_instances[instance_id] = TinygradInstance(
                    instance_id=instance_id,
                    shard_assignments=shard_assignments,
                )
            else:
                if command.sharding != Sharding.Pipeline:
                    raise ValueError(
                        "multi-node Tinygrad requires Sharding.Pipeline "
                        f"(got {command.sharding})"
                    )
                ephemeral_port = random_ephemeral_port()
                hosts_by_node = get_mlx_ring_hosts_by_node(
                    selected_cycle=selected_cycle,
                    cycle_digraph=cycle_digraph,
                    ephemeral_port=ephemeral_port,
                    node_network=node_network,
                )
                target_instances[instance_id] = TinygradInstance(
                    instance_id=instance_id,
                    shard_assignments=shard_assignments,
                    hosts_by_node=hosts_by_node,
                    ephemeral_port=ephemeral_port,
                )
```

Add `Sharding` to the imports at top of file if not already present.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest src/exo/master/tests/test_placement.py -v`
Expected: PASS (including the two new tests)

- [ ] **Step 5: Type check + lint**

Run: `uv run basedpyright src/exo/master/placement.py && uv run ruff check src/exo/master/placement.py`
Expected: 0 errors, 0 warnings

- [ ] **Step 6: Commit**

```bash
git add src/exo/master/placement.py src/exo/master/tests/test_placement.py
git commit -m "feat(master): populate pipeline transport fields for multi-node tinygrad"
```

---

## Task 2: Create pipeline_group.py with framing and connect logic

**Files:**
- Create: `src/exo/worker/engines/tinygrad/pipeline_group.py`
- Test: `src/exo/worker/engines/tinygrad/tests/test_pipeline_group.py`

### Design

Wire format (same socket carries both directions):
- 5-byte header: `<tag:u8><length:u32-le>`
- `tag=0` (HIDDEN): payload = raw fp16 tensor bytes, preceded by 8-byte shape (`<batch:u16><seq_len:u32><hidden_dim:u16>`) then raw data
- `tag=1` (TOKEN): payload = `<token_id:i32-le><stop:u8>` (5 bytes)
- `tag=2` (STOP): payload = empty (0 bytes)

Ring topology: rank k listens on `ephemeral_port`, waits for rank `(k-1) % world_size` to connect (inbound). After accepting, rank k opens outbound connection to rank `(k+1) % world_size` at that node's host. Bootstrap deadlock-free by having all ranks start the listener, then each opens the outbound connection with retries.

- [ ] **Step 1: Write failing unit test** at `src/exo/worker/engines/tinygrad/tests/test_pipeline_group.py`

```python
import pytest
import numpy as np
from exo.worker.engines.tinygrad.pipeline_group import PipelineGroup, _encode_hidden, _decode_hidden, _encode_token, _decode_token

def test_encode_decode_token_roundtrip():
    buf = _encode_token(42, stop=True)
    token, stop = _decode_token(buf)
    assert token == 42 and stop is True

def test_encode_decode_hidden_roundtrip():
    arr = np.arange(12, dtype=np.float16).reshape(1, 3, 4)
    buf = _encode_hidden(arr)
    out = _decode_hidden(buf)
    assert out.shape == arr.shape
    assert np.array_equal(out, arr)

@pytest.mark.asyncio
async def test_two_rank_ring_localhost():
    # rank 0 on 127.0.0.1:PORT_A, rank 1 on 127.0.0.1:PORT_B
    # hosts_by_node maps node_id -> list of Host(host, port)
    # For testing we'll use localhost and two distinct ports.
    import threading
    from exo.shared.types.common import Host, NodeId

    node0 = NodeId("node0")
    node1 = NodeId("node1")
    port = 42000  # pick fixed test port per test file
    hosts = {
        node0: [Host(host="127.0.0.1", port=port)],
        node1: [Host(host="127.0.0.1", port=port + 1)],
    }

    groups: list[PipelineGroup | None] = [None, None]

    def run_rank(rank: int):
        g = PipelineGroup.connect(
            rank=rank,
            world_size=2,
            hosts_by_node={node0: hosts[node0], node1: hosts[node1]},
            bind_port=port + rank,
            node_rank_mapping=[node0, node1],
        )
        groups[rank] = g

    t0 = threading.Thread(target=run_rank, args=(0,))
    t1 = threading.Thread(target=run_rank, args=(1,))
    t0.start(); t1.start()
    t0.join(timeout=5); t1.join(timeout=5)

    assert groups[0] is not None
    assert groups[1] is not None
    assert groups[0].rank == 0
    assert groups[1].rank == 1
    # round-trip a token rank 1 -> rank 0
    groups[1].send_token(99, stop=False)
    token, stop = groups[0].recv_token()
    assert token == 99 and not stop
    groups[0].close(); groups[1].close()
```

- [ ] **Step 2: Run tests — expect fail**

Run: `uv run pytest src/exo/worker/engines/tinygrad/tests/test_pipeline_group.py -v`
Expected: FAIL with ImportError (module doesn't exist yet)

- [ ] **Step 3: Implement `src/exo/worker/engines/tinygrad/pipeline_group.py`**

```python
"""TCP-ring pipeline transport for tinygrad multi-node sharding.

Each rank listens on `bind_port`, waits for rank (k-1) mod N to connect
(receive direction), then opens an outbound connection to rank (k+1) mod N
(send direction). Messages are 5-byte tag+length framed; payloads carry
either hidden-state tensors (fp16) or sampled tokens.
"""
import socket
import struct
import time
from dataclasses import dataclass
from typing import Self

import numpy as np
from loguru import logger

from exo.shared.types.common import Host, NodeId

TAG_HIDDEN = 0
TAG_TOKEN = 1
TAG_STOP = 2

_HEADER = struct.Struct("<BI")          # tag, length
_HIDDEN_SHAPE = struct.Struct("<HIH")   # batch, seq_len, hidden_dim
_TOKEN_PAYLOAD = struct.Struct("<iB")   # token_id, stop


def _encode_hidden(arr: np.ndarray) -> bytes:
    assert arr.dtype == np.float16 and arr.ndim == 3
    batch, seq_len, hidden_dim = arr.shape
    shape_hdr = _HIDDEN_SHAPE.pack(batch, seq_len, hidden_dim)
    return shape_hdr + arr.tobytes()


def _decode_hidden(buf: bytes) -> np.ndarray:
    batch, seq_len, hidden_dim = _HIDDEN_SHAPE.unpack_from(buf, 0)
    data = np.frombuffer(buf[_HIDDEN_SHAPE.size:], dtype=np.float16)
    return data.reshape(batch, seq_len, hidden_dim)


def _encode_token(token_id: int, stop: bool) -> bytes:
    return _TOKEN_PAYLOAD.pack(token_id, 1 if stop else 0)


def _decode_token(buf: bytes) -> tuple[int, bool]:
    token_id, stop = _TOKEN_PAYLOAD.unpack_from(buf, 0)
    return int(token_id), bool(stop)


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
    tag, length = _HEADER.unpack(header)
    payload = _recv_exactly(sock, length) if length else b""
    return tag, payload


@dataclass
class PipelineGroup:
    rank: int
    world_size: int
    recv_sock: socket.socket  # from rank-1
    send_sock: socket.socket  # to rank+1

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

        next_rank = (rank + 1) % world_size
        next_node = node_rank_mapping[next_rank]
        next_hosts = hosts_by_node[next_node]
        if not next_hosts:
            raise RuntimeError(f"no hosts for next rank {next_rank} ({next_node})")
        next_host = next_hosts[0]

        listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen.bind(("0.0.0.0", bind_port))
        listen.listen(1)
        logger.info(f"pipeline rank {rank}: listening on :{bind_port}")

        # Open outbound connection (to next rank) with retry — other rank may
        # still be setting up its listener.
        deadline = time.monotonic() + connect_timeout
        send_sock: socket.socket | None = None
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((next_host.host, next_host.port))
                send_sock = s
                break
            except (ConnectionRefusedError, OSError) as e:
                last_err = e
                time.sleep(retry_interval)
        if send_sock is None:
            listen.close()
            raise RuntimeError(
                f"pipeline rank {rank}: could not connect to rank {next_rank} "
                f"at {next_host.host}:{next_host.port}: {last_err}"
            )
        logger.info(
            f"pipeline rank {rank}: connected to rank {next_rank} "
            f"at {next_host.host}:{next_host.port}"
        )

        recv_sock, _addr = listen.accept()
        listen.close()
        logger.info(f"pipeline rank {rank}: accepted connection from rank-1")

        # Disable Nagle — forward latency matters more than bandwidth.
        for s in (send_sock, recv_sock):
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        return cls(
            rank=rank,
            world_size=world_size,
            recv_sock=recv_sock,
            send_sock=send_sock,
        )

    def send_hidden(self, arr: np.ndarray) -> None:
        _send_frame(self.send_sock, TAG_HIDDEN, _encode_hidden(arr))

    def recv_hidden(self) -> np.ndarray:
        tag, payload = _recv_frame(self.recv_sock)
        if tag != TAG_HIDDEN:
            raise RuntimeError(f"expected HIDDEN, got tag={tag}")
        return _decode_hidden(payload)

    def send_token(self, token_id: int, stop: bool) -> None:
        _send_frame(self.send_sock, TAG_TOKEN, _encode_token(token_id, stop))

    def recv_token(self) -> tuple[int, bool]:
        tag, payload = _recv_frame(self.recv_sock)
        if tag != TAG_TOKEN:
            raise RuntimeError(f"expected TOKEN, got tag={tag}")
        return _decode_token(payload)

    def send_stop(self) -> None:
        _send_frame(self.send_sock, TAG_STOP, b"")

    def recv_any(self) -> tuple[int, bytes]:
        return _recv_frame(self.recv_sock)

    def close(self) -> None:
        for s in (self.recv_sock, self.send_sock):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest src/exo/worker/engines/tinygrad/tests/test_pipeline_group.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Type check + lint**

Run: `uv run basedpyright src/exo/worker/engines/tinygrad/pipeline_group.py && uv run ruff check src/exo/worker/engines/tinygrad/pipeline_group.py`
Expected: 0 errors

- [ ] **Step 6: Commit**

```bash
git add src/exo/worker/engines/tinygrad/pipeline_group.py src/exo/worker/engines/tinygrad/tests/test_pipeline_group.py
git commit -m "feat(tinygrad): add TCP-ring pipeline transport"
```

---

## Task 3: Make weight_loader gate boundary layers on rank position

**Files:**
- Modify: `src/exo/worker/engines/tinygrad/weight_loader.py` — add `is_first_rank: bool`, `is_last_rank: bool` to `load_transformer_weights`

Currently the weight loader reads `embed_tokens`, `lm_head`, and `final_norm` unconditionally. For pipeline sharding, only rank 0 needs `embed_tokens`; only the last rank needs `lm_head` + `final_norm`. Loading all three on every rank wastes VRAM (embed and lm_head can each be hundreds of MB on bigger models).

- [ ] **Step 1: Read current `load_transformer_weights` signature**

Run: `grep -n "^def load_transformer_weights\|embed_tokens\|lm_head\|final_norm" src/exo/worker/engines/tinygrad/weight_loader.py`

Note the current parameter list; confirm how `TransformerWeights` is constructed.

- [ ] **Step 2: Write failing test** in `src/exo/worker/engines/tinygrad/tests/test_weight_loader.py` (create if absent)

Verify that passing `is_first_rank=False` leaves `embed_tokens` placeholder (or `None` variant) without erroring, and that `is_last_rank=False` leaves `lm_head` / `final_norm` similarly. Use a tiny fake safetensors fixture from existing tests if available; otherwise use a mocked `safe_load`.

```python
def test_weight_loader_skips_embed_on_non_first_rank(tmp_path, monkeypatch):
    # minimal fake: write a safetensors file with only layers 8..16
    # then load with start_layer=8, end_layer=16, is_first_rank=False, is_last_rank=False
    # assert weights.embed_tokens is None and weights.lm_head is None and weights.final_norm is None
    ...
```

- [ ] **Step 3: Make `TransformerWeights` fields optional for boundary items**

Modify `TransformerWeights` NamedTuple in `weight_loader.py` so `embed_tokens: EmbedWeight | None`, `lm_head: LinearWeight | None`, `final_norm: Tensor | None`.

- [ ] **Step 4: Update `load_transformer_weights` signature** to accept `is_first_rank: bool = True, is_last_rank: bool = True` and skip loading the respective keys when false.

- [ ] **Step 5: Run tests**

Run: `uv run pytest src/exo/worker/engines/tinygrad/tests/test_weight_loader.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/exo/worker/engines/tinygrad/weight_loader.py src/exo/worker/engines/tinygrad/tests/test_weight_loader.py
git commit -m "feat(tinygrad): make boundary weights optional for pipeline shards"
```

---

## Task 4: Make forward_pass pipeline-aware

**Files:**
- Modify: `src/exo/worker/engines/tinygrad/forward.py`

Current signature:

```python
def forward_pass(
    weights: TransformerWeights,
    input_ids: Tensor,
    cache: KVCache | None,
    position_offset: int | Tensor = 0,
    rope_cos: Tensor | None = None,
    rope_sin: Tensor | None = None,
) -> tuple[Tensor, KVCache]:
```

New behaviour:
- If `weights.embed_tokens is None`, `input_ids` is actually a pre-embedded hidden state `Tensor[batch, seq_len, hidden_dim]`. Skip embedding.
- Run layers `start_layer..end_layer` as today. (Already shaped by how weights are loaded; no change needed to layer loop.)
- If `weights.final_norm is None`, skip final norm.
- If `weights.lm_head is None`, return the hidden state instead of logits. Caller handles this based on rank.

- [ ] **Step 1: Change signature** to accept `Tensor` input that may be hidden state:

```python
def forward_pass(
    weights: TransformerWeights,
    input_or_hidden: Tensor,
    cache: KVCache | None,
    position_offset: int | Tensor = 0,
    rope_cos: Tensor | None = None,
    rope_sin: Tensor | None = None,
) -> tuple[Tensor, KVCache]:
    if weights.embed_tokens is not None:
        x = apply_embedding(weights.embed_tokens, input_or_hidden)
    else:
        x = input_or_hidden  # already embedded by rank 0
    ...
    # layer loop unchanged
    ...
    if weights.final_norm is not None:
        x = rms_norm(x, weights.final_norm, eps=weights.config.rms_norm_eps)
    if weights.lm_head is not None:
        logits = apply_lm_head(weights.lm_head, x)
        return logits, cache
    return x, cache  # hidden state when not last rank
```

- [ ] **Step 2: Update callers** in `generate.py`, `generator/generate.py`, and any tests that assume logits are always returned.

- [ ] **Step 3: Type check + existing tests**

Run: `uv run basedpyright src/exo/worker/engines/tinygrad/forward.py && uv run pytest src/exo/worker/engines/tinygrad/tests/ -v`
Expected: PASS (may need test updates; keep existing single-rank behaviour identical when all boundaries are present)

- [ ] **Step 4: Commit**

```bash
git add src/exo/worker/engines/tinygrad/forward.py src/exo/worker/engines/tinygrad/generator/generate.py
git commit -m "feat(tinygrad): make forward_pass skip embed/norm/lm_head per rank"
```

---

## Task 5: Add ConnectToGroup handler to tinygrad runner

**Files:**
- Modify: `src/exo/worker/runner/llm_inference/tinygrad_runner.py`

The runner currently crashes on `ConnectToGroup` (falls into the default `case _:`). Add a case that constructs a `PipelineGroup` and stashes it. For single-node instances (`hosts_by_node is None`), skip and stay idle (the planner will only emit `ConnectToGroup` for multi-node instances).

- [ ] **Step 1: Import required types at top of `tinygrad_runner.py`**

```python
from exo.shared.types.tasks import ConnectToGroup
from exo.shared.types.worker.instances import TinygradInstance
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    ...
)
from exo.worker.engines.tinygrad.pipeline_group import PipelineGroup
```

(Check `runners.py` — `RunnerConnected`/`RunnerConnecting` already exist for MLX. Reuse them.)

- [ ] **Step 2: Add `group: PipelineGroup | None = None` as a local after `runner_id` is defined.**

- [ ] **Step 3: Add new case before the `case _:` catch-all**

```python
case ConnectToGroup() if isinstance(current_status, (RunnerIdle, RunnerFailed)):
    instance = bound_instance.instance
    assert isinstance(instance, TinygradInstance)
    assert instance.hosts_by_node is not None, (
        "ConnectToGroup received but TinygradInstance has no hosts_by_node"
    )
    assert instance.ephemeral_port is not None
    current_status = RunnerConnecting()
    event_sender.send(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
    )
    event_sender.send(TaskAcknowledged(task_id=task.task_id))

    # Build rank -> node mapping from shard_assignments.
    shard_assignments = instance.shard_assignments
    node_rank_mapping: list[NodeId | None] = [None] * shard_metadata.world_size
    for rid, shard in shard_assignments.runner_to_shard.items():
        node_id = shard_assignments.runner_to_node[rid]
        node_rank_mapping[shard.device_rank] = node_id
    assert all(n is not None for n in node_rank_mapping)

    group = PipelineGroup.connect(
        rank=shard_metadata.device_rank,
        world_size=shard_metadata.world_size,
        hosts_by_node=instance.hosts_by_node,
        bind_port=instance.ephemeral_port,
        node_rank_mapping=[n for n in node_rank_mapping if n is not None],
    )
    current_status = RunnerConnected()
    logger.info(f"tinygrad pipeline group connected at rank {group.rank}")
```

- [ ] **Step 4: Gate `LoadModel` case** to accept `RunnerConnected` as a valid prior state (in addition to `RunnerIdle`/`RunnerFailed`). Pass `group` into `initialize_tinygrad` and `load_tinygrad_items` so the shard only loads layers in its range.

- [ ] **Step 5: Thread `group` through to `warmup_inference` and `tinygrad_generate`**

See Task 6 for the generator-side changes; in the runner, pass `group=group` whenever calling them.

- [ ] **Step 6: Add cleanup on shutdown**

In the `Shutdown()` case:

```python
if group is not None:
    group.close()
    group = None
```

- [ ] **Step 7: Run existing runner tests**

Run: `uv run pytest src/exo/worker/tests/ -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/exo/worker/runner/llm_inference/tinygrad_runner.py
git commit -m "feat(tinygrad-runner): handle ConnectToGroup for pipeline sharding"
```

---

## Task 6: Pipeline-aware generate.py

**Files:**
- Modify: `src/exo/worker/engines/tinygrad/generator/generate.py`
- Modify: `src/exo/worker/engines/tinygrad/utils_tinygrad.py` — accept `group` in `initialize_tinygrad`

The main generator function gains a `group: PipelineGroup | None` parameter. When `group is None or group.world_size == 1`, existing single-node behaviour runs unchanged. Otherwise dispatch by rank.

- [ ] **Step 1: Add `group` parameter to `tinygrad_generate` and `warmup_inference`**

```python
def tinygrad_generate(
    model: TransformerWeights,
    tokenizer: Any,
    task: TextGenerationTaskParams,
    prompt: str,
    kv_prefix_cache: Any = None,
    on_prefill_progress: Callable[[int, int], None] | None = None,
    group: "PipelineGroup | None" = None,
) -> Generator[GenerationResponse]:
    if group is None or group.world_size == 1:
        yield from _single_rank_generate(model, tokenizer, task, prompt)
        return
    if group.rank == 0:
        yield from _rank0_pipeline_generate(model, tokenizer, task, prompt, group)
    else:
        _worker_pipeline_loop(model, group, is_last=(group.rank == group.world_size - 1))
```

Refactor the current body of `tinygrad_generate` into `_single_rank_generate` (no change in behaviour).

- [ ] **Step 2: Implement `_rank0_pipeline_generate`**

```python
def _rank0_pipeline_generate(
    model: TransformerWeights, tokenizer, task, prompt, group: PipelineGroup,
) -> Generator[GenerationResponse]:
    input_ids = _encode_prompt(tokenizer, prompt)
    prompt_tokens = len(input_ids)
    input_ids = _pad_to_bucket(input_ids)

    # Local prefill on rank 0's layer slice (embed + first slice).
    config = model.config
    cache = KVCache(
        num_layers=len(model.layers),
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seq_len=min(config.max_position_embeddings, 4096),
    )
    for i in range(len(model.layers)):
        cache.keys[i] = cache.keys[i].contiguous().realize()
        cache.values[i] = cache.values[i].contiguous().realize()

    prompt_tensor = Tensor(input_ids, dtype=dtypes.int32).reshape(1, -1).contiguous().realize()
    with Context(BEAM=0):
        hidden, _ = forward_pass(
            model, prompt_tensor, cache,
            position_offset=0,
            rope_cos=model.rope_cos, rope_sin=model.rope_sin,
        )
    # Ship prefill hidden to next rank.
    group.send_hidden(_tensor_to_np(hidden))
    # Wait for last rank to produce the first sampled token.
    token_id, stop = group.recv_token()
    position = prompt_tokens

    max_tokens = task.max_output_tokens or DEFAULT_MAX_TOKENS
    for token_idx in range(max_tokens):
        token_text = tokenizer.decode([token_id])
        eos_ids = _get_eos_ids(tokenizer, model.config)
        is_eos = token_id in eos_ids or stop
        finish_reason = "stop" if is_eos else ("length" if token_idx == max_tokens - 1 else None)

        # (stats/usage construction same as single-rank)
        yield GenerationResponse(
            text="" if is_eos else token_text,
            token=token_id,
            logprob=None, top_logprobs=None,
            finish_reason=finish_reason, stats=None, usage=None,
        )
        if finish_reason is not None:
            # Signal downstream ranks to stop.
            group.send_stop()
            break

        # Decode one token: embed -> first slice -> ship hidden -> recv next token.
        input_tensor = Tensor([[token_id]], dtype=dtypes.int32).contiguous().realize()
        with Context(BEAM=0):
            hidden, _ = forward_pass(
                model, input_tensor, cache,
                position_offset=position,
                rope_cos=model.rope_cos, rope_sin=model.rope_sin,
            )
        group.send_hidden(_tensor_to_np(hidden))
        token_id, stop = group.recv_token()
        position += 1
```

- [ ] **Step 3: Implement `_worker_pipeline_loop`**

```python
def _worker_pipeline_loop(
    model: TransformerWeights, group: PipelineGroup, is_last: bool,
) -> None:
    config = model.config
    cache = KVCache(
        num_layers=len(model.layers),
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seq_len=min(config.max_position_embeddings, 4096),
    )
    for i in range(len(model.layers)):
        cache.keys[i] = cache.keys[i].contiguous().realize()
        cache.values[i] = cache.values[i].contiguous().realize()

    position: int = 0  # updated based on first hidden's seq_len
    first = True
    while True:
        tag, payload = group.recv_any()
        if tag == TAG_STOP:
            group.send_stop()  # propagate
            return
        assert tag == TAG_HIDDEN
        hidden_np = _decode_hidden(payload)
        hidden = Tensor(hidden_np).contiguous().realize()

        offset: int | Tensor = 0 if first else position
        first = False
        with Context(BEAM=0):
            out, _ = forward_pass(
                model, hidden, cache,
                position_offset=offset,
                rope_cos=model.rope_cos, rope_sin=model.rope_sin,
            )
        if is_last:
            # out is logits [1, seq_len, vocab]; sample from the last position.
            result = sample_token(
                out[:, -1:, :], temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P, top_logprobs_count=0, request_logprobs=False,
            )
            group.send_token(result.token_id, stop=False)
        else:
            group.send_hidden(_tensor_to_np(out))
        position += int(hidden_np.shape[1])
```

- [ ] **Step 4: Add `_tensor_to_np` helper**

```python
def _tensor_to_np(t: Tensor) -> np.ndarray:
    return t.numpy().astype(np.float16)
```

- [ ] **Step 5: Update `warmup_inference(model, tokenizer, group=None)` to accept group and defer to rank-specific paths.** For a 5-token warmup, run normal `tinygrad_generate` — since all ranks are in the same generator, warmup will exercise the pipeline path naturally.

- [ ] **Step 6: Run pipeline unit tests**

Run: `uv run pytest src/exo/worker/engines/tinygrad/tests/test_pipeline_group.py src/exo/worker/engines/tinygrad/tests/ -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/exo/worker/engines/tinygrad/generator/generate.py src/exo/worker/engines/tinygrad/utils_tinygrad.py
git commit -m "feat(tinygrad): pipeline-aware generate with per-rank dispatch"
```

---

## Task 7: Two-process end-to-end integration test

**Files:**
- Create: `src/exo/worker/engines/tinygrad/tests/test_pipeline_integration.py`

Spin up two processes on localhost, each loading a tiny test model split at layer boundary. Drive a prompt through and verify rank 0 yields tokens.

- [ ] **Step 1: Find or generate a tiny-model fixture**

Look for existing tinygrad test fixtures — there may be a stub Llama-style model with 2-4 layers. If absent, use Llama-3.2-1B-Instruct-4bit (cached on both boxes) and split at layer 8 of 16.

- [ ] **Step 2: Implement the integration test**

```python
import multiprocessing as mp
import pytest

def _rank_process(rank: int, world_size: int, port_base: int, model_path: str, queue: mp.Queue):
    # Each process connects to the ring and runs one prompt, placing resulting
    # tokens on the queue from rank 0.
    ...

@pytest.mark.slow
@pytest.mark.skipif("EXO_TEST_TINYGRAD_PIPELINE not in os.environ")
def test_two_rank_pipeline_end_to_end(tmp_path):
    mp.set_start_method("spawn", force=True)
    q: mp.Queue = mp.Queue()
    procs = [mp.Process(target=_rank_process, args=(r, 2, 42100, str(tmp_path), q)) for r in range(2)]
    for p in procs: p.start()
    tokens = [q.get(timeout=60)]
    for p in procs: p.join(timeout=30)
    assert len(tokens) > 0
```

Mark as slow / skip unless env var set — don't run by default in CI.

- [ ] **Step 3: Run manually on beast with a real model**

Run: `EXO_TEST_TINYGRAD_PIPELINE=1 uv run pytest src/exo/worker/engines/tinygrad/tests/test_pipeline_integration.py -v -s`
Expected: PASS, produces at least one token

- [ ] **Step 4: Commit**

```bash
git add src/exo/worker/engines/tinygrad/tests/test_pipeline_integration.py
git commit -m "test(tinygrad): two-rank pipeline end-to-end integration test"
```

---

## Task 8: Live two-machine smoke test

Manual verification step — not scripted since it requires both beast and cube.

- [ ] **Step 1: Sync the branch to cube**

On beast:
```bash
git push origin feature/linux-support
```

On cube:
```bash
cd ~/dev/exo-linux && git pull --ff-only
```

- [ ] **Step 2: Start exo on both machines**

On beast (with existing launcher):
```bash
~/exo-run.sh
```

On cube:
```bash
cd ~/dev/exo-linux
CUDA_PATH=$(pwd)/.venv/lib/python3.13/site-packages/nvidia/cuda_runtime uv run exo
```

- [ ] **Step 3: Via dashboard, place a model larger than either single GPU's VRAM**

E.g. a 16–26 GB model. Choose the placement preview that spans both nodes with `Sharding.Pipeline`.

- [ ] **Step 4: Verify**

- `nvidia-smi` on BOTH machines shows an exo runner process with non-trivial VRAM usage
- Dashboard shows model "Ready"
- A chat completion returns tokens

If any of these fail, systematic-debugging skill → read logs on both sides → return to Task 5/6 to fix.

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Multi-node tinygrad placement populates transport fields (Task 1)
- ✅ TCP ring transport with token + hidden framing (Task 2)
- ✅ Weight loader avoids loading embed/lm_head/norm where not needed (Task 3)
- ✅ forward_pass skips embed/norm/lm_head per rank (Task 4)
- ✅ Runner handles ConnectToGroup (Task 5)
- ✅ Generate dispatches per-rank (Task 6)
- ✅ End-to-end test (Task 7) + live test (Task 8)

**Known gaps (accepted):**
- No tensor parallelism across nodes (by design — plan explicitly restricts multi-node to Pipeline sharding)
- No JIT across pipeline boundaries on rank-0 decode loop (decode fallback uses the non-JIT prefill-style path inside `Context(BEAM=0)`) — first pass prioritises correctness; JIT can be re-added in a follow-up
- Token broadcast latency adds one network RTT per decode step — acceptable on a LAN (~0.1–0.5 ms)
- No backpressure / pipelining of decode steps — strictly sequential
- Error recovery: peer crash = reconnect is not implemented; runner exits and supervisor restarts

**Out of scope:**
- Image model pipeline sharding (`image_models/runner.py` — same strategy applies but separate plan)
- Shard-aware model download (each node still downloads the full model)
