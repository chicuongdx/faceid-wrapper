# FaceID Wrapper

## Description

This is a wrapper for the FaceID API. It provides a simple interface to the FaceID API.

The wrapper provides the following functionality:
- Register a new user
- Verify a user
- Identify a user from an image or video frame

I use fastAPI to create the API and the requests library to make the requests to the FaceID API.

## Model

For AI solutions:
- First, I use Yolov5 to detect the face in the image or video frame.
- Then, I use the InceptionResnetV1 with weights vggface2 for extract the face embedding.

## Installation

To install the wrapper, you can use pip:

```bash
git clone https://github.com/chicuongdx/faceid-wrapper.git
cd faceid-wrapper
pip install requirments.txt
```

Thank you visit my project! Have a nice day!