# Copyright 2022 The DLRover Authors. All rights reserved.
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
from unittest import mock

from dlrover.python.common.constants import (
    NodeExitReason,
    NodeType,
    RendezvousName,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.node import Node
from dlrover.python.master.dist_master import DistributedJobMaster
from dlrover.python.master.node.event_callback import (
    AllReduceNodeHandlingCallback,
)
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.tests.test_utils import MockK8sPSJobArgs, mock_k8s_client

_dlrover_ctx = Context.singleton_instance()


class AllReduceEventCallbackTest(unittest.TestCase):
    def setUp(self):
        mock_k8s_client()
        params = MockK8sPSJobArgs()
        params.initilize()
        self.master = DistributedJobMaster(2222, params)
        self.master.job_manager.start_auto_scaling = mock.MagicMock(
            return_value=True
        )
        self.master.job_manager.all_critical_node_completed = mock.MagicMock(
            return_value=True
        )
        self.event_cb = AllReduceNodeHandlingCallback(self.master)
        self.job_context = get_job_context()

    def tearDown(self):
        self.job_context.clear_job_nodes()
        self.job_context.request_stop()

    def test_on_node_started(self):
        worker = Node(node_type=NodeType.WORKER, node_id=0)
        self.event_cb.on_node_started(worker, None)
        et_manager = self.master.rdzv_managers[RendezvousName.TRAINING]
        self.assertEqual(len(et_manager._alive_nodes), 1)
        self.event_cb.on_node_deleted(worker, None)
        self.assertEqual(len(et_manager._alive_nodes), 0)

    def test_on_node_succeeded(self):
        worker = Node(
            node_type=NodeType.WORKER,
            node_id=0,
            critical=True,
        )
        self.event_cb.on_node_succeeded(worker, None)

    def test_on_node_failed(self):
        worker = Node(
            node_type=NodeType.WORKER,
            node_id=0,
            critical=True,
        )
        worker.max_relaunch_count = 2
        worker.relaunch_count = 1
        worker.exit_reason = NodeExitReason.FATAL_ERROR
        self.event_cb.on_node_failed(worker, None)
        self.assertEqual(self.event_cb._job_context.get_failed_node_cnt(), 1)
        _dlrover_ctx.relaunch_always = True
        self.event_cb.on_node_failed(worker, None)
        worker.relaunch_count = 2
        self.event_cb.on_node_failed(worker, None)
        _dlrover_ctx.relaunch_always = False

        self.event_cb._failed_worker_count = worker.max_relaunch_count
        self.event_cb._stop_job_if_needed(worker)

        self.event_cb._available_worker_num = 4
        self.event_cb._min_node = 8
        self.event_cb._stop_job_if_needed(worker)
