# Image Encryptor/Decryptor
[Web App](https://samueluy-image-encryptor-digimap-main-qsy762.streamlit.app/)\
This implementation of the Unimodular Encryption using Logistic Maps algorithm is adapted from the code written by Indra Bayu Muktyas, Sulistiawati, Samsul Arifin and published on his [GitHub repository](https://github.com/muktyas/encryption-unimodular-logistic-map) in 2020.

## DIGIMAP - Group 14
- Deculawan, Ryan Jay
- Tan, Lance Griffin
- Uy, Samuel Jedidiah
- Yu, Ethan Angelo

## Introduction
This is an Image Encryptor/Decryptor tool that allows users to protect their images by encrypting them with a password. The tool is implemented using the Unimodular Encryption algorithm and logistic maps. It provides options to encrypt and decrypt images and supports downloading the encrypted or decrypted image.

### Usage
1. Upload the image you want to encrypt or decrypt.
2. Select a key from the drop-down menu or enter your own password.
3. Choose whether you want to encrypt or decrypt the image.
4. View the encrypted or decrypted image.
5. Download the encrypted or decrypted image for safekeeping.

Please note that the maximum password length is 14 digits.

## Code
The code provided includes the following functions:

- `convert_image(img)`: Converts an image to bytes.
- `run(col1, col2, image_path, option)`: Runs the encryption or decryption process.
- `main()`: The main function that sets up the web app and handles user interactions.
- `log_map(x0, n)`: Generates a pseudo-random sequence of numbers using the logistic map equation.
- `elementary_row_operation(m, row_i, row_j, r)`: Performs an elementary row operation on a matrix.
- `swap(m, row_i, row_j)`: Swaps two rows in a matrix.
- `generate_key(n, x0)`: Generates a key matrix and its inverse for encryption and decryption.
- `encrypt(image_path, size, x0)`: Encrypts an image using a given key.
- `decrypt(gb, size, x0)`: Decrypts an image using a given key.
- `check_same(original_image, decrypted_image)`: Compares two images to determine if they are the same.
- `factors(n)`: Returns a list of all factors of a given integer.
