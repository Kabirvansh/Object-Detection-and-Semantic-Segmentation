import numpy as np
import torch
import torch.nn as nn
import tqdm
import pandas as pd
from PIL import Image as im
from torchvision.transforms import ToPILImage

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=11):  
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)
        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    def __call__(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            #Sine(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            #Sine(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )
        return contract
    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            #Sine(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            #Sine(),
            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
            )    
        return expand 

def detect_and_segment(images):
    """
    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """
    N = images.shape[0]

    # Placeholder for predictions
    pred_class = np.zeros((N, 2), dtype=np.int32)
    pred_bboxes = np.zeros((N, 2, 4), dtype=np.float64)
    pred_seg = np.zeros((N, 4096), dtype=np.int32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    yolo_model.to(device)
    yolo_model.eval()

    unet_model = UNET(in_channels=3, out_channels=11).to(device)
    unet_model.load_state_dict(torch.load('unet_best.pth', map_location=device))
    unet_model.eval()

    to_pil = ToPILImage()

    for i in tqdm.tqdm(range(len(images))):
        data_tensor = torch.tensor(images[i].reshape(64, 64, 3), dtype=torch.uint8).permute(2, 0, 1).to(device)
        data = to_pil(data_tensor)

        results = yolo_model(data, size=256)
        predictions = results.pandas().xyxy[0]

        predictions['confidence'] = pd.to_numeric(predictions['confidence'], errors='coerce')    #used stack overflows and chatgpt to debug the dtype and ntype error
        if predictions.empty:   
            pred_class[i] = np.zeros(2, dtype=np.int32)  
            pred_bboxes[i] = np.zeros((2, 4), dtype=np.float64)  
        else:
            top_predictions = predictions.nlargest(2, 'confidence')
        classes = top_predictions['class'].values
        boxes = top_predictions[['xmin', 'ymin', 'xmax', 'ymax']].values

        indices = np.argsort(classes)
        pred_class[i] = classes[indices]
        pred_bboxes[i] = boxes[indices]

        notmal_tensor = data_tensor.float() / 255.0
        with torch.no_grad():
            output = unet_model(notmal_tensor.unsqueeze(0)) 

        segmentation_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        final = segmentation_mask 

        pred_seg[i] = final.flatten()

    return pred_class, pred_bboxes, pred_seg
