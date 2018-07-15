# insightocr





### Text Recognition Accuracy on Chinese dataset by [caffe-ocr](https://github.com/senlinuc/caffe_ocr)

| Network   | LSTM | 4x1 Pooling | Gray | Test Acc |
| --------- | ---- | ----------- | ---- | -------- |
| SimpleNet | N    | Y           | Y    | 99.37%   |
| SE-ResNet34 | N    | Y           | Y    | 99.73%   |


### Text Recognition Accuracy on [VGG_Text](http://www.robots.ox.ac.uk/~vgg/data/text/), on the subset of label size<=18

| Network   | LSTM | 4x1 Pooling | Gray | Test Acc |
| --------- | ---- | ----------- | ---- | -------- |
| SimpleNet | Y    | Y           | Y    | 87.17%  |
| SE-ResNet34 | Y    | Y           | Y    | TODO  |
| SE-ResNet50-PReLU | Y    | Y           | Y    | TODO  |
