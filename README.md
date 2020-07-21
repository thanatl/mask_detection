# mask_detection
Mask detection model  using Pytorch

Compose_data.py used data from </br>
1. https://github.com/prajnasb/observations 
2. https://www.kaggle.com/alexandralorenzo/maskdetection 

Extract both file in 'data' folder 
 - with_mask
 - without_mask
 - yolo

 Then, execute ``compose_data.py`` to create json file contain bounding box (left x, top y, width, height) and label </br>
 Initialize dataloader using ``MaskDataLoader`` </br>

Model, Loss function, and other utility functions are modified from source: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection </br>
