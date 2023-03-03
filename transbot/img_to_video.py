import cv2
import os
from time import time

def merge_image_to_video(fold_path: str, img_size: tuple = (640, 480), fps: int = 20, fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), save_path: str = "output.mp4"):
    """合成图片为视频

    Args:
        fold_path (str): 图片文件夹的路径
        img_size (tuple, optional): 图片的尺寸，需要fold_path下的图片统一尺寸, cv2.VideoWriter的参数. Defaults to (640, 480).
        fps (int, optional): 帧率, cv2.VideoWriter的参数 . Defaults to 20.
        fourcc (cv2.VideoWriter_fourcc, optional): cv2.VideoWriter的参数. Defaults to cv2.VideoWriter_fourcc('m', 'p', '4', 'v').
        save_path (str, optional): 视频保存的路径，cv2.VideoWriter的参数. Defaults to "output.mp4".
    """
    start_time = time()
    video = cv2.VideoWriter(save_path, fourcc, fps, img_size)

    file_ls = []
    for f1 in os.listdir(fold_path):
        file = os.path.join(fold_path, f1)
        file_ls.append(file)
    # 按照文件路径split("\\").split("_")[0]的值进行排序
    file_ls.sort(key=lambda x: int(x.split("\\")[-1].split("_")[0]))
    print(file_ls)
    img_ls = map(cv2.imread, file_ls)

    print("merge_image_to_video_io: ", time() - start_time)
    for i in img_ls:
        video.write(i)

    video.release()
    print("merge_image_to_video: ", time() - start_time)



if __name__ == '__main__':
    path = r"data\traindatava"
    merge_image_to_video(path, save_path="output.mp4")
   