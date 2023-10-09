import cv2
import numpy as np
import torch
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from cloths_segmentation.pre_trained_models import create_model
import matplotlib.pyplot as plt

model = create_model("Unet_2020-10-30")
model.eval()


user_pic=input("enter the picture:- ")

image = load_rgb(user_pic)


transform = albu.Compose([albu.Normalize(p=1)], p=1)
padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)

x = transform(image=padded_image)["image"]
x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

with torch.no_grad():
    prediction = model(x)[0][0]

mask = (prediction > 0).cpu().numpy().astype(np.uint8)

mask = unpad(mask, pads)



dst = cv2.addWeighted(image, 1, (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(np.uint8), 0.5, 0)


dst_bgr = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)

# Save the dst image as "output.jpg"
cv2.imwrite("output.jpg", dst_bgr)
