# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter server."""

from concurrent import futures
import logging
import numpy as np

import grpc
from concurrent import futures
import grpc
import numpy as np

from v3 import image_service_pb2_grpc
from v3 import image_service_pb2

class ImageDetectorServicer(image_service_pb2_grpc.ImageDetectorServicer):
    def DetectImage(self, request, context):
        # Extract the image data from the request message
        image_data = np.frombuffer(request.image_data, dtype=np.uint8)
        # TODO: Process the image data and generate the response
        response = image_service_pb2.Response()
        response.results.extend(['result 1', 'result 2', 'result 3'])
        return response

def serve():
    port = '50051'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    image_service_pb2_grpc.add_ImageDetectorServicer_to_server(ImageDetectorServicer(), server)
    server.add_insecure_port('[::]:' + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
