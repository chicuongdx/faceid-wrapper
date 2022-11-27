from video_frame import VideoFrame
import cv2

if __name__ == '__main__':
    cap_url = 0
    model_url = 'models/best.pt'
    video_frame = VideoFrame(cap_url, model_url)
    video_frame.run()