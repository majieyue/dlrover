# Copyright 2026 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
from typing import Dict, List, Optional

from dlrover.python.common.constants import (
    NodeGroupStrategy,
    NodeType,
)
from dlrover.python.common.log import default_logger as logger


class SoftGroupSchedule(object):
    """Schedule of unequally-sized node groups (--soft-group-affinity).

    Fully isolated from the plain ``--group-affinity`` path
    (:class:`NodeGroupSchedule`/:func:`validate_topology`/
    :func:`resolve_group_id` in ``dlrover.python.master.resource.job``):
    a soft schedule declares a heterogeneous per-group pod count and
    resolves every worker's group id at pod-creation time so the scaler
    labels pods with ``scheduling/rack-group`` for segment-affinity
    scheduling. There is deliberately NO equal-size constraint, but
    EVERY group size must be a multiple of the EP slot
    (``ep_workers = EP/R`` pods): this EP alignment is an unconditional
    start-up requirement, not an option.

    Args:
        strategy: one of :class:`NodeGroupStrategy` (contiguous or
            ep_pp_dp), consulted by :func:`resolve_soft_group_id`.
        sizes: ``{group_id: pod count}``, e.g. ``{0: 30, 1: 20}``; the
            keys are the node-group ids and can be any non-negative
            integers (the fill order is ascending) and every count must
            be a multiple of the EP slot.
        tp / pp / ep / cp: Megatron parallel sizes. Only ``tp == 1`` and
            ``cp == 1`` are supported today.
        num_nodes: total number of worker nodes (``N``), must equal the
            sum of ``sizes``.
        ranks_per_node: number of ranks per worker node (``R``).
        no_group_failover: when true (--no-group-failover), a relaunched
            (FO) worker drops its node-group labels so it can be
            scheduled onto any segment (see
            DistributedJobManager._relaunch_node); when false (default)
            the relaunch keeps the group labels and follows the
            original segment affinity.
    """

    def __init__(
        self,
        strategy: str,
        sizes: Dict[int, int],
        tp: int = 1,
        pp: int = 1,
        ep: int = 1,
        cp: int = 1,
        num_nodes: int = 0,
        ranks_per_node: int = 0,
        no_group_failover: bool = False,
    ):
        self.strategy = strategy
        self.sizes = sizes
        self.tp = tp
        self.pp = pp
        self.ep = ep
        self.cp = cp
        self.num_nodes = num_nodes
        self.ranks_per_node = ranks_per_node
        self.no_group_failover = no_group_failover
        # Laid-out {rank_index: group_id} for the ep_pp_dp strategy,
        # built lazily by resolve_soft_group_id.
        self._ep_pp_dp_layout: Optional[List[int]] = None
        self._layout_lock = threading.Lock()


def ep_group_workers(schedule: SoftGroupSchedule) -> int:
    """Return the EP slot size in pods (``EP/R``, one EP communication
    group of ``EP`` consecutive ranks). The caller must have validated
    ``EP % R == 0`` (see validate_soft_group_topology)."""
    if schedule.ranks_per_node <= 0:
        return 0
    return schedule.ep // schedule.ranks_per_node


def validate_soft_group_topology(
    schedule: Optional[SoftGroupSchedule],
) -> None:
    """Validate a soft group schedule (``--soft-group-affinity``).

    Distinct from :func:`validate_topology`: there is no equal-size
    constraint and no segment-count divisibility; raising ``ValueError``
    leaves the schedule unapplied so the master fails to start cleanly.

    Constraints (``N`` = #worker nodes, ``R`` = ranks/node, ``G`` =
    ``len(sizes)``, ``model_parallel = TP*PP*CP``,
    ``dense_dp = N*R/model_parallel``, ``ep_workers = EP/R``):

      - ``sizes`` is non-empty with positive counts and non-negative keys;
      - ``sum(sizes) == N``;
      - scope: ``TP == 1`` and ``CP == 1``; ``PP > 0`` and ``EP > 0``;
      - ``N % model_parallel == 0`` and ``dense_dp % EP == 0`` so the
        data-parallel size ``dp = dense_dp/EP`` is an integral number;
      - ``EP % R == 0`` (an EP slot is composed of whole nodes) and every
        group size is a multiple of ``ep_workers`` — the EP alignment is
        mandatory by default, no opt-in flag;
      - when the strategy is ``ep_pp_dp``: additionally
        ``DP_nodes % ep_workers == 0`` so every pipeline stage holds a
        whole number of EP slots.
    """
    if schedule is None or not schedule.sizes:
        return

    sizes = schedule.sizes
    tp, pp, ep, cp = schedule.tp, schedule.pp, schedule.ep, schedule.cp
    n, r = schedule.num_nodes, schedule.ranks_per_node

    total = sum(sizes.values())
    if total != n:
        raise ValueError(
            "soft-group-affinity sizes sum to "
            f"{total} pods, but the job declares N={n} worker replicas; "
            "align --soft-group-affinity with the worker count."
        )
    if any(size <= 0 for size in sizes.values()):
        raise ValueError(
            f"soft-group-affinity group sizes must be positive, got {sizes}."
        )
    if any(key < 0 for key in sizes.keys()):
        raise ValueError(
            "soft-group-affinity group ids must be non-negative, got "
            f"{sorted(sizes.keys())}."
        )

    if tp != 1:
        raise ValueError(
            "--soft-group-affinity currently requires TP=1 "
            f"(got tp={tp}); TP>1 support is planned for a later phase."
        )
    if cp != 1:
        raise ValueError(
            "--soft-group-affinity currently requires CP=1 "
            f"(got cp={cp}); CP>1 support is planned for a later phase."
        )
    if pp <= 0 or ep <= 0:
        raise ValueError(
            f"--soft-group-affinity requires pp={pp} and ep={ep} to be "
            "positive."
        )
    if r <= 0:
        raise ValueError(
            "--soft-group-affinity requires ranks_per_node>0 "
            f"(NodeResource.gpu_num), got {r}."
        )

    model_parallel = tp * pp * cp
    if n % model_parallel != 0:
        raise ValueError(
            "--soft-group-affinity requires the worker node count N="
            f"{n} to be divisible by TP*PP*CP={model_parallel}."
        )
    dense_dp = (n * r) // model_parallel
    if dense_dp % ep != 0:
        raise ValueError(
            "--soft-group-affinity requires dense_dp="
            f"{dense_dp} (N*R/(TP*PP*CP)) to be divisible by EP={ep} so "
            "the data-parallel size dp=dense_dp/EP is an integer."
        )

    # EP alignment is the default, not an option: an EP slot must be
    # well defined (EP % R == 0) and every group must be a whole number
    # of EP slots, otherwise the master fails to start.
    if ep % r != 0:
        raise ValueError(
            "--soft-group-affinity requires EP="
            f"{ep} to be divisible by ranks_per_node R={r} so an EP "
            "group (slot) is composed of whole nodes."
        )
    ep_workers = ep // r
    for group_id, size in sizes.items():
        if size % ep_workers != 0:
            raise ValueError(
                "--soft-group-affinity requires every group size to be a "
                f"multiple of the EP slot (ep_workers={ep_workers}); "
                f"group {group_id} has {size} pods."
            )

    if schedule.strategy == NodeGroupStrategy.EP_PP_DP:
        dp_nodes = n // model_parallel
        if dp_nodes % ep_workers != 0:
            raise ValueError(
                "--soft-group-affinity with node-group-strategy=ep_pp_dp "
                f"requires DP_nodes={dp_nodes} (nodes per pipeline stage) "
                f"to be divisible by ep_workers={ep_workers}."
            )


def _ep_pp_dp_layout(schedule: SoftGroupSchedule) -> List[int]:
    """Compute (once per schedule) the full-world group sequence of the
    ``ep_pp_dp`` strategy: ``{rank_index: group_id}`` for ``k = 0..N-1``.

    Each round takes one EP slot (``ep_workers`` consecutive pods) from
    every non-exhausted group in ascending group-id order — "take
    ep-size pods from a group, jump to the next group, and so on".
    Validation guarantees every size is a whole number of EP slots, so
    the round robin consumes exactly all pods; the leftover sweep below
    is a defensive fallback for schedules built bypassing validation.
    """
    with schedule._layout_lock:
        if schedule._ep_pp_dp_layout is not None:
            return schedule._ep_pp_dp_layout
        sizes = schedule.sizes
        total = sum(sizes.values())
        slot = ep_group_workers(schedule)
        remaining = {g: sizes[g] for g in sorted(sizes) if sizes[g] > 0}
        layout: List[int] = []

        # Round robin of whole EP slots: one slot per group per round.
        progressed = True
        while progressed:
            progressed = False
            for group_id in list(remaining.keys()):
                if remaining[group_id] >= slot:
                    layout.extend([group_id] * slot)
                    remaining[group_id] -= slot
                    progressed = True

        # Defensive: complete with leftover pods, one per group per sweep.
        # Unreachable for validated schedules (every size is a multiple
        # of the EP slot).
        if any(remaining.values()):
            logger.warning(
                "The ep_pp_dp soft group layout found leftover pods %s "
                "outside whole EP slots (the schedule bypassed the EP "
                "alignment validation); they are placed one per group per "
                "sweep and their EP groups may straddle groups.",
                {g: c for g, c in remaining.items() if c > 0},
            )
            while len(layout) < total and any(remaining.values()):
                for group_id in list(remaining.keys()):
                    if remaining[group_id] <= 0 or len(layout) >= total:
                        continue
                    layout.append(group_id)
                    remaining[group_id] -= 1

        if len(layout) != total:
            raise ValueError(
                "soft-group-affinity failed to lay out all worker pods; "
                f"expected {total}, got {len(layout)}."
            )

        straddled = [
            start
            for start in range(0, total, slot)
            if len(set(layout[start : start + slot])) > 1
        ]
        if straddled:
            logger.warning(
                "The ep_pp_dp soft group layout contains %d EP slot(s) "
                "crossing groups (sizes %s), starting at pod %s.",
                len(straddled),
                sizes,
                straddled[:5],
            )
        else:
            logger.info(
                "The ep_pp_dp soft group layout of sizes %s keeps every "
                "EP slot inside one group.",
                sizes,
            )
        schedule._ep_pp_dp_layout = layout
        return layout


def resolve_soft_group_id(
    schedule: Optional[SoftGroupSchedule],
    node_type: str,
    rank_index: int,
) -> Optional[int]:
    """Resolve the soft node group id of a worker by its rank index.

    - ``contiguous``: groups occupy cumulative contiguous rank ranges in
      ascending group-id order (fill up one group then the next).
    - ``ep_pp_dp``: the global round-robin EP-slot layout (see
      :func:`_ep_pp_dp_layout`).

    Returns ``None`` for non-worker nodes, when no soft schedule is set,
    or when the rank falls beyond the declared world (e.g. a scale-up
    beyond ``sum(sizes)``), leaving the pod unlabeled.
    """
    if schedule is None or not schedule.sizes:
        return None
    if node_type != NodeType.WORKER:
        return None
    total = sum(schedule.sizes.values())
    if rank_index < 0 or rank_index >= total:
        return None
    if schedule.strategy == NodeGroupStrategy.EP_PP_DP:
        return _ep_pp_dp_layout(schedule)[rank_index]

    start = 0
    for group_id in sorted(schedule.sizes.keys()):
        size = schedule.sizes[group_id]
        if start <= rank_index < start + size:
            return group_id
        start += size
    return None
