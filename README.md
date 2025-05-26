# Bird recognition system
The aim of this project was to develop and evaluate a system for recognizing bird species based on recordings of their songs, with a particular focus on the impact of geographical location on classification performance. To achieve this, two classification models were developed:
- One model is based on deep neural networks, which is responsible for species classification using spectrogram representations of bird sounds.
- Another one uses kernel density estimation to perform classification based on species occurrence coordinates and to model their geographical distribution at the same time.
These models were integrated into a multi-classifier system to take advantage of their complementary strengths.

In the scripts folder, you’ll find a complete, ready-to-use system that integrates both models and includes a user interface for bird classification.
