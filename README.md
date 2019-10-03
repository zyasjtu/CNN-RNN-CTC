# CNN + RNN + CTC Loss for OCR

This is a tensorflow re-implementation for the paper: "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition"

More details: https://arxiv.org/pdf/1507.05717.pdf

# Dependencies
* Python3
* tensorflow
* numpy
* opencv-python

Dependencies can be installed with
```bash
pip install -r requirements.txt
```

# Data Preparation
* Put all images in ./data/number_img/
* Supply a file to specify the image names and corresponding gt

For example: number_list
```bash
a_1.jpg 54420196
a_2.jpg 8862
```

# Train
```bash
python train.py
```

# Valid
```bash
python valid.py
```