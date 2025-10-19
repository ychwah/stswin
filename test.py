import cv2
from baseline import build_text_feature_stack

img = cv2.imread("input.jpg")
feats = build_text_feature_stack(img, rtv_lambda=0.02)

x3 = feats["OCRPrime3"]   # HxWx3 float32 [0..1]
# feed x3 to your transformer backbone (normalize as usual)

# If you can handle 4 channels, prefer this for robustness:
x4 = feats["OCRStack4"]   # HxWx4 float32 [0..1]
