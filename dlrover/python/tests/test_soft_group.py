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

import unittest

from dlrover.python.common.constants import NodeGroupStrategy, NodeType
from dlrover.python.master.resource.job import (
    NodeGroupSchedule,
    resolve_group_id,
    validate_topology,
)
from dlrover.python.master.resource.soft_group import (
    SoftGroupSchedule,
    resolve_soft_group_id,
    validate_soft_group_topology,
)


def _soft(
    strategy=NodeGroupStrategy.CONTIGUOUS,
    sizes=None,
    tp=1,
    pp=1,
    ep=1,
    cp=1,
    num_nodes=None,
    ranks_per_node=8,
    no_group_failover=False,
):
    if sizes is None:
        sizes = {0: 1}
    return SoftGroupSchedule(
        strategy=strategy,
        sizes=sizes,
        tp=tp,
        pp=pp,
        ep=ep,
        cp=cp,
        num_nodes=(
            num_nodes if num_nodes is not None else sum(sizes.values())
        ),
        ranks_per_node=ranks_per_node,
        no_group_failover=no_group_failover,
    )


class SoftGroupScheduleTest(unittest.TestCase):
    def test_carry_fields(self):
        schedule = _soft(
            strategy=NodeGroupStrategy.EP_PP_DP,
            sizes={0: 30, 1: 20},
            pp=2,
            ep=40,
            no_group_failover=True,
        )
        self.assertEqual(schedule.sizes, {0: 30, 1: 20})
        self.assertEqual(
            (schedule.tp, schedule.pp, schedule.ep, schedule.cp), (1, 2, 40, 1)
        )
        self.assertEqual(
            (schedule.num_nodes, schedule.ranks_per_node), (50, 8)
        )
        self.assertTrue(schedule.no_group_failover)
        self.assertIsNone(schedule._ep_pp_dp_layout)


class ValidateSoftGroupTopologyTest(unittest.TestCase):
    def test_none_or_empty_is_noop(self):
        validate_soft_group_topology(None)
        validate_soft_group_topology(_soft(sizes={}))

    def test_ok_unequal(self):
        # The whole point of the soft path: heterogeneous sizes are legal
        # (EP slot = 1 pod with ep=8, R=8, so any size is aligned).
        validate_soft_group_topology(_soft(sizes={0: 30, 1: 20}, pp=2, ep=8))

    def test_ok_ep_slot_aligned(self):
        # N=64, R=8, EP=16, PP=2 -> dense_dp=256, dp=16, ep_workers=2;
        # every size is a multiple of 2. EP alignment is the default, no
        # flag involved.
        validate_soft_group_topology(
            _soft(
                strategy=NodeGroupStrategy.EP_PP_DP,
                sizes={0: 32, 1: 32},
                pp=2,
                ep=16,
            )
        )
        # {0:30, 1:20} with ep=40: dp=5 and ep_workers=5 divides both sizes.
        validate_soft_group_topology(
            _soft(
                sizes={0: 30, 1: 20},
                pp=2,
                ep=40,
            )
        )

    def test_sum_must_equal_num_nodes(self):
        with self.assertRaisesRegex(ValueError, "sum to 50"):
            validate_soft_group_topology(
                _soft(sizes={0: 30, 1: 20}, num_nodes=49, pp=2, ep=8)
            )

    def test_positive_sizes_and_keys(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_soft_group_topology(_soft(sizes={0: 0, 1: 4}, pp=1, ep=1))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_soft_group_topology(
                _soft(sizes={-1: 4, 0: 4}, pp=1, ep=1)
            )

    def test_tp_cp_scope(self):
        with self.assertRaisesRegex(ValueError, "TP=1"):
            validate_soft_group_topology(
                _soft(sizes={0: 8, 1: 8}, tp=2, pp=2, ep=8)
            )
        with self.assertRaisesRegex(ValueError, "CP=1"):
            validate_soft_group_topology(
                _soft(sizes={0: 8, 1: 8}, cp=2, pp=2, ep=8)
            )

    def test_invalid_parallel_sizes(self):
        with self.assertRaisesRegex(ValueError, "pp=0"):
            validate_soft_group_topology(_soft(sizes={0: 8, 1: 8}, pp=0, ep=8))
        with self.assertRaisesRegex(ValueError, "ep=0"):
            validate_soft_group_topology(_soft(sizes={0: 8, 1: 8}, pp=2, ep=0))
        with self.assertRaisesRegex(ValueError, "ranks_per_node"):
            validate_soft_group_topology(
                _soft(sizes={0: 8, 1: 8}, pp=2, ep=8, ranks_per_node=0)
            )

    def test_num_nodes_must_divide_model_parallel(self):
        with self.assertRaisesRegex(ValueError, r"TP\*PP\*CP=3"):
            validate_soft_group_topology(_soft(sizes={0: 8, 1: 8}, pp=3, ep=8))

    def test_dp_must_be_integral(self):
        # N=50, R=8, PP=2 -> dense_dp=200 is not divisible by EP=16:
        # not even a valid Megatron shape (dp=12.5).
        with self.assertRaisesRegex(ValueError, "divisible by EP"):
            validate_soft_group_topology(
                _soft(sizes={0: 30, 1: 20}, pp=2, ep=16)
            )

    def test_ep_pp_dp_requires_ep_multiple_of_r(self):
        # N=24, R=8, PP=2 -> dense_dp=96 divides EP=12 (dp integral) but
        # EP=12 is not a multiple of R=8, so an EP slot is not whole
        # nodes.
        with self.assertRaisesRegex(
            ValueError, "EP=12 to be divisible by ranks_per_node"
        ):
            validate_soft_group_topology(
                _soft(
                    strategy=NodeGroupStrategy.EP_PP_DP,
                    sizes={0: 12, 1: 12},
                    pp=2,
                    ep=12,
                )
            )

    def test_alignment_is_default_not_optin(self):
        # dense_dp=256 with pp=2, EP=16 -> dp integral and EP%R==0, but
        # 31 is odd so the group is not a multiple of the EP slot
        # (ep_workers=2). No flag exists to relax this: the master fails
        # to start for BOTH strategies.
        for strategy in (
            NodeGroupStrategy.CONTIGUOUS,
            NodeGroupStrategy.EP_PP_DP,
        ):
            with self.assertRaisesRegex(ValueError, "group 0 has 31"):
                validate_soft_group_topology(
                    _soft(
                        strategy=strategy,
                        sizes={0: 31, 1: 33},
                        pp=2,
                        ep=16,
                    )
                )

    def test_alignment_requires_ep_multiple_of_r_even_contiguous(self):
        # The EP slot must be well-defined regardless of the strategy:
        # N=48, R=8, PP=2 -> dense_dp=192 divides EP=12, but EP=12 is
        # not a multiple of R=8.
        with self.assertRaisesRegex(ValueError, "EP=12 to be divisible"):
            validate_soft_group_topology(
                _soft(
                    sizes={0: 33, 1: 15},
                    pp=2,
                    ep=12,
                )
            )


class ResolveSoftGroupIdTest(unittest.TestCase):
    def test_none_or_non_worker_returns_none(self):
        self.assertIsNone(resolve_soft_group_id(None, NodeType.WORKER, 0))
        self.assertIsNone(resolve_soft_group_id(_soft(sizes={}), "worker", 0))
        schedule = _soft(sizes={0: 4, 1: 4})
        self.assertIsNone(resolve_soft_group_id(schedule, "ps", 0))

    def test_contiguous_unequal_sizes(self):
        schedule = _soft(sizes={0: 30, 1: 20})
        result = [
            resolve_soft_group_id(schedule, NodeType.WORKER, rank)
            for rank in (0, 29, 30, 49, 50)
        ]
        self.assertEqual(result, [0, 0, 1, 1, None])

    def test_contiguous_uses_ascending_group_keys(self):
        schedule = _soft(sizes={5: 2, 2: 3})
        result = [
            resolve_soft_group_id(schedule, NodeType.WORKER, rank)
            for rank in range(5)
        ]
        self.assertEqual(result, [2, 2, 2, 5, 5])

    def test_ep_pp_dp_equal_sizes_slot_round_robin(self):
        # N=16, R=8, PP=2, EP=16 -> dp_nodes=8, ep_workers=2: one EP slot
        # (2 pods) from every group per round.
        schedule = _soft(
            strategy=NodeGroupStrategy.EP_PP_DP,
            sizes={0: 8, 1: 8},
            pp=2,
            ep=16,
        )
        self.assertEqual(
            [
                resolve_soft_group_id(schedule, NodeType.WORKER, rank)
                for rank in range(16)
            ],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        )

    def test_ep_pp_dp_leftover_pods_tolerated_with_warning(self):
        # Defensive resolver behavior for a schedule that bypassed the
        # (now mandatory) EP alignment validation: {0:9, 1:7} has 1+1
        # leftover pods after the whole slots; they complete the layout
        # pod-by-pod, so the final EP slot straddles both groups —
        # reported with a WARNING, never a resolver failure.
        schedule = _soft(
            strategy=NodeGroupStrategy.EP_PP_DP,
            sizes={0: 9, 1: 7},
            pp=2,
            ep=16,
        )
        with self.assertLogs("dlrover.logger", level="WARNING") as logs:
            layout = [
                resolve_soft_group_id(schedule, NodeType.WORKER, rank)
                for rank in range(16)
            ]
        self.assertEqual(
            layout,
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
        )
        self.assertTrue(any("crossing groups" in log for log in logs.output))

    def test_ep_pp_dp_layout_built_once(self):
        schedule = _soft(
            strategy=NodeGroupStrategy.EP_PP_DP,
            sizes={0: 8, 1: 8},
            pp=2,
            ep=16,
        )
        first = resolve_soft_group_id(schedule, NodeType.WORKER, 3)
        layout = schedule._ep_pp_dp_layout
        second = resolve_soft_group_id(schedule, NodeType.WORKER, 3)
        self.assertEqual(first, second)
        self.assertIs(layout, schedule._ep_pp_dp_layout)

    def test_ep_pp_dp_equal_sizes_matches_stripe_formula(self):
        # When every group holds exactly one EP slot per pipeline stage
        # (size = PP*EP/R, i.e. N*R = PP*EP*G) the round-robin layout
        # reproduces the static ep_pp_dp stripe formula exactly, for
        # contiguous AND opaque (non 0-based, per #1745) group ids.
        N, R, PP, EP, G = 16, 8, 2, 16, 4
        for ga in (
            {i: N // G for i in range(G)},  # contiguous ids {0,1,2,3}
            {5: 4, 9: 4, 17: 4, 23: 4},  # opaque ids
        ):
            static = NodeGroupSchedule(
                strategy=NodeGroupStrategy.EP_PP_DP,
                tp=1,
                pp=PP,
                ep=EP,
                cp=1,
                num_nodes=N,
                ranks_per_node=R,
            )
            validate_topology(ga, static)
            schedule = _soft(
                strategy=NodeGroupStrategy.EP_PP_DP,
                sizes=ga,
                pp=PP,
                ep=EP,
            )
            self.assertEqual(
                [
                    resolve_soft_group_id(schedule, NodeType.WORKER, k)
                    for k in range(N)
                ],
                [
                    resolve_group_id(ga, NodeType.WORKER, k, static)
                    for k in range(N)
                ],
            )

    def test_ep_pp_dp_invariants_9216_equal_sizes(self):
        # N=1152 = 9 segments x 128 workers, R=8, PP=16, EP=32:
        # ep_workers=4, DP_nodes=72 (2 rounds over the 9 groups per
        # stage). EP slots stay inside one group and the same
        # within-stage position always maps to the same group.
        N, R, PP, EP, G = 1152, 8, 16, 32, 9
        schedule = _soft(
            strategy=NodeGroupStrategy.EP_PP_DP,
            sizes={g: N // G for g in range(G)},
            pp=PP,
            ep=EP,
        )
        layout = [
            resolve_soft_group_id(schedule, NodeType.WORKER, rank)
            for rank in range(N)
        ]
        dp_nodes, slot = N // PP, EP // R
        for start in range(0, N, slot):
            self.assertEqual(
                len(set(layout[start : start + slot])),
                1,
                f"EP slot at [{start},{start + slot})",
            )
        for position in range(dp_nodes):
            self.assertEqual(
                len({layout[pp * dp_nodes + position] for pp in range(PP)}),
                1,
                f"PP column {position}",
            )
        sums = {g: layout.count(g) for g in range(G)}
        self.assertEqual(sums, {g: N // G for g in range(G)})


if __name__ == "__main__":
    unittest.main()
