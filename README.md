# SteganoGAN Project

## Overview
This project implements a steganographic method for hiding secret texts within images using Generative Adversarial Networks (GAN). The primary goal is to embed secret messages into images in such a way that the resulting images (Stegano-Images) appear realistic and unsuspicious to third parties. The GAN used in this project is a Deep Convolutional GAN (DCGAN), and the combined embedding and GAN process is referred to as SteganoGAN.

## Project Purpose
Image Steganography allows for the discreet transmission of secret messages by embedding them into images. Unlike cryptography, where the existence of the encoded data is known, steganography aims to hide the very presence of the data. This project aims to develop an algorithm that can seamlessly embed messages into images, making it difficult for third parties to detect the presence of hidden messages.

## Results
The final model achieved an accuracy of 99.999% for message retrieval, with the generated Stegano-Images appearing indistinguishable from the original cover images. The project successfully demonstrated the feasibility of using GANs for image steganography.

## Training and Testing
The training sequence involved training the decoder, SteganoGAN, discriminator, and GAN in that order. The testing phase involved using a completely new image and message to verify the model's performance. The model successfully retrieved the hidden message with 100% accuracy.

## How to Run
1. Preprocess the raw images by running `data_process.py`.
2. Train the SteganoGAN model by running `train.py`.
3. Test the model by running `test.py`.

## Dependencies
- TensorFlow
- Keras
- OpenCV
- NumPy

## Files
- `data_process.py`: Preprocesses raw images and saves them as numpy arrays.
- `decode.py`: Decodes the hidden message from the Stegano-Image.
- `encode.py`: Encodes the secret message into a format suitable for embedding.
- `decoder.py`: Defines the decoder model for extracting hidden messages.
- `discriminator.py`: Defines the discriminator model for the GAN.
- `gan.py`: Defines the GAN model combining the generator and discriminator.
- `generator.py`: Defines the generator model for creating Stegano-Images.
- `steganogan.py`: Defines the SteganoGAN model combining the generator and decoder.
- `test.py`: Tests the model by embedding and extracting a sample message.
- `train.py`: Trains the SteganoGAN model using the preprocessed data.

## Acknowledgements
This project is based on the principles of steganography and deep learning, combining cyber security and GANs to achieve high accuracy in message embedding and retrieval.