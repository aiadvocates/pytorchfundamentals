

# Audio Classification with PyTorch

In this learn module we will be dive into how to do audio classification with PyTorch. There are multiple ways to build an audio classification model. You can use the waveform, tag the wav file, or use computer vision on the spectorgram image. In this tutorial we will first break down how to understand audio data, then we will build the model using transfer learning and computer vision on the spectorgram images.

The dataset we will be using is the open dataset [Speech Commands]() which is built into PyTorch [datasets]().

"Each utterance is stored as a one-second (or less)
WAVE format file, with the sample data encoded as
linear 16-bit single-channel PCM values, at a 16 KHz
rate. There are 2,618 speakers recorded, each with
a unique eight-digit hexadecimal identifier assigned."

# Prerequisites