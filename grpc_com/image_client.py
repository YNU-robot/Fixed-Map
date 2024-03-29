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
"""The Python implementation of the GRPC helloworld.Greeter client."""

from __future__ import print_function

import logging

import grpc
from v3 import image_service_pb2
from v3 import image_service_pb2_grpc
import numpy as np
import time


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    print("Will try to greet world ...")
    with grpc.insecure_channel('100.77.102.113:50005') as channel:
        stub = image_service_pb2_grpc.ImageDetectorStub(channel)
        # load image
        image = np.random.randint(0, 255, size=(640, 480, 3), dtype=np.uint8)
        # convert the image to a byte sequence and create the request message
        start_time = time.time()
        image_data = image.tobytes()
        request = image_service_pb2.Detect(image_data=image_data)

        # Call the DetectImage method and print the response
        response = stub.DetectImage(request)
        print(response.results)
        print('time:', time.time() - start_time)



if __name__ == '__main__':
    # logging.basicConfig()
    run()
