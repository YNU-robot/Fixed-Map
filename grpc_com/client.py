from Dxr_grpc.dxr_grpc_client import Dxr_grpc_client
import cv2
import time
import numpy as np
 
# 初始化客户端，ip和端口号
dxr_client = Dxr_grpc_client('100.77.102.113', "50004")
 
json_data = {
    'type': 'face',
}


image = np.random.randint(0, 255, size=(640, 480, 3), dtype=np.uint8)
 
start_time = time.time()
# 调用get_response方法，获取处理后的图像
res = dxr_client.get_response(image, json_data)
for i in res:
    print(i)
print('time:', time.time() - start_time)