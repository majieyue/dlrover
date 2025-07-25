# Copyright 2025 The DLRover Authors. All rights reserved.
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
import time

from dlrover.python.unified.common.failure import FailureDesc
from dlrover.python.unified.tests.base import BaseTest


class FailureDescTest(BaseTest):
    def test_basic(self):
        desc = FailureDesc(
            workload_name="test",
            failure_time=int(time.time()),
            failure_level=1,
            reason="unknown",
        )
        self.assertEqual(desc.failure_obj, "WORKLOAD")
        self.assertEqual(desc.workload_name, "test")
        self.assertEqual(desc.failure_level, 1)
