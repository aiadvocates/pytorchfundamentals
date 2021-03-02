

# Audio Classification with PyTorch


Ever wonder how the voice assistance actually work? How do they understand the words that we say? When you think about voice assitants you have the first step which is speech to text, then the nlp step which is the word embedding (turning words into numbers), then you have a classification of the utterance (what people say) to the intent (what they want the voice assitant to do) If you are following this learning path, you will have learned how the the NLP part happens. Now we want to look at how we get the text from the spoken audio. Of course audio classification can be used for many things, not just speech assistances. For example in music you can classify genres, or detect illness by the tone in someones voice, and probably even more applications that we havent even thought of yet.

In this learn module we will be dive into how to do audio classification with PyTorch. There are multiple ways to build an audio classification model. You can use the waveform, tag sections of a wav file, or even use computer vision on the spectorgram image. In this tutorial we will first break down how to understand audio data, from analog to digital representations, then we will build the model using computer vision on the spectorgram images. Thats right, you can turn audio into an image representation and then do computer vision to classify the word!

What we are focusing on in this is how the actualy wave file can be understood. We will be using a very simple model that can understand yes and no. The dataset we will be using is the open dataset [Speech Commands](https://pytorch.org/audio/stable/datasets.html#speechcommands) which is built into PyTorch [datasets](https://pytorch.org/audio/stable/datasets.html). This dataset has 36 total different words/sounds to be used for classification. Each utterance is stored as a one-second (or less) WAVE format file. We will only be using `yes` and `no` for a binary classification. 

# Learning objectives

# Prerequisites

- Knowledge of Python