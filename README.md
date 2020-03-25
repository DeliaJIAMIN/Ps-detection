# Ps-detection
from FAL detection


# 1.Install packages
  `pip install -r requirements.txt `

# 2.Download model weights
  `Run bash weights/download_weights.sh`

# 3.Global classifer
  `python global_classifier.py --input_path examples/modified.jpg --model_path weights/global.pth`

# 4.Local Detector
  `python local_detector.py --input_path examples/modified.jpg --model_path weights/local.pth --dest_folder out/`


Note: Our models are trained on faces cropped by the dlib CNN face detector. Although in both scripts we included the --no_crop option to run the models without face crops, it is used for images with already cropped faces.

Evaluate the dataset, run:

# 5.Download the dataset
```Download the dataset
cd data
bash download_valset.sh
cd ..
# Run evaluation script. Model weights need to be downloaded.
python eval.py --dataroot data --global_pth weights/global.pth --local_pth weights/local.pth --gpu_id 0```
