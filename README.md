# RegSeg

A clean and flexible implementation of "Rethink Dilated Convolution for Real-time Semantic Segmentation".

## Usage

clone this repo:

```shell
git clone https://github.com/Asthestarsfalll/RegSeg
```

install requirements:

```shell
pip install -r requirements.txt
```

### Train

Define your training config in /configs folder, whcih contains the default config of regseg_exp48_decoder_26.

Define your model config in /models/config.py.

```shell
python train.py --config path/to/your.config
```

The training logs will be saved in /training_log folder by default and the tensorboard files will be saved in /tensorboard folder.

### Test

Currently not supported.

## Result

| model                   | batch_size | dataset    | mIoU  |
| ----------------------- | ---------- | ---------- | ----- |
| regseg_exp48_decoder_26 | 8          | cityscapes | 77.59 |
| regseg_exp48_decoder_26 | 8          | cityscapes | 77.81 |
| regseg_exp48_decoder_26 | 32         | cityscapes | 77.14 |
|                         |            |            |       |

