# Copyright 2024 The DLRover Authors. All rights reserved.
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

# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""

import grpc
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2

from . import brain_pb2 as brain__pb2


class BrainStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.persist_metrics = channel.unary_unary(
            "/brain.Brain/persist_metrics",
            request_serializer=brain__pb2.JobMetrics.SerializeToString,
            response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
        )
        self.optimize = channel.unary_unary(
            "/brain.Brain/optimize",
            request_serializer=brain__pb2.OptimizeRequest.SerializeToString,
            response_deserializer=brain__pb2.OptimizeResponse.FromString,
        )
        self.get_job_metrics = channel.unary_unary(
            "/brain.Brain/get_job_metrics",
            request_serializer=brain__pb2.JobMetricsRequest.SerializeToString,
            response_deserializer=brain__pb2.JobMetricsResponse.FromString,
        )


class BrainServicer(object):
    """Missing associated documentation comment in .proto file."""

    def persist_metrics(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def optimize(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def get_job_metrics(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_BrainServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "persist_metrics": grpc.unary_unary_rpc_method_handler(
            servicer.persist_metrics,
            request_deserializer=brain__pb2.JobMetrics.FromString,
            response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
        ),
        "optimize": grpc.unary_unary_rpc_method_handler(
            servicer.optimize,
            request_deserializer=brain__pb2.OptimizeRequest.FromString,
            response_serializer=brain__pb2.OptimizeResponse.SerializeToString,
        ),
        "get_job_metrics": grpc.unary_unary_rpc_method_handler(
            servicer.get_job_metrics,
            request_deserializer=brain__pb2.JobMetricsRequest.FromString,
            response_serializer=brain__pb2.JobMetricsResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "brain.Brain", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class Brain(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def persist_metrics(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/brain.Brain/persist_metrics",
            brain__pb2.JobMetrics.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def optimize(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/brain.Brain/optimize",
            brain__pb2.OptimizeRequest.SerializeToString,
            brain__pb2.OptimizeResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def get_job_metrics(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/brain.Brain/get_job_metrics",
            brain__pb2.JobMetricsRequest.SerializeToString,
            brain__pb2.JobMetricsResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
