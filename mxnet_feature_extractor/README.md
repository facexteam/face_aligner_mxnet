# Caffe Feature Extractor
A wrapper for extractring features from Caffe network, with a config file to define network parameter.

## Python requirements:
```
mxnet
numpy
json
```

## extractor config example
Images are load by opencv (cv2.imread).

the config file might looks like:

```json
{
    "network_model": "C:/zyf/dnn_models/face_models/insight-face/model-r34-amf/model,0",
    "feature_layer": "fc1_output",
    "batch_size": 4,
    "input_width": 112,
    "input_height": 112,
    "image_as_grey": 0,
    "channel_swap": "2, 1, 0",
    "data_mean": "",
    "input_scale": 1.0,
    "mirror_trick": 1,
    "normalize_output": 1,
    "cpu_only": 0,
    "gpu_id": 0
}
```

Note:

 *"feature_layer"*: from which layer to extract features, e.g. "fc1_output";

 *"batch_size"* in the config json file would overwrite the "batch size" in the prototxt;

 *"input_width"*, *"input_height"*: input data shape to bind for model;

 *"image_as_grey"*: read image in grey model;

 *"channel_swap"*: how to change the channel orders, default order is "BGR" as in OpenCV, "2, 1, 0" will change into "RGB";

 *"mirror_trick"*: =0, original features; =1, eltsum(original, mirrored)/2; =2, eltmax(original, mirrored);

 *"normalize_output"*: =1, will do L2-normalization before output; =0, no normalization.

 *"cpu_only"*: =1, caffe in CPU mode; =0, caffe in GPU mode.

 *"gpu_id"*: which GPU to use when cpu_only==0.
