

# Audio Classification with PyTorch

In this learn module we will be dive into how to do audio classification with PyTorch. There are multiple ways to build an audio classification model. You can use the waveform, tag the wav file, or use computer vision on the spectorgram image. In this tutorial we will first break down how to understand audio data, then we will build the model using computer vision on the spectorgram images.

The dataset we will be using is the open dataset [Speech Commands](https://pytorch.org/audio/stable/datasets.html#speechcommands) which is built into PyTorch [datasets](https://pytorch.org/audio/stable/datasets.html). This dataset has 36 total different words/sounds to be used for classification. Each utterance is stored as a one-second (or less) WAVE format file. We will only be using `yes` and `no` for a binary classification. 

# Prerequisites

