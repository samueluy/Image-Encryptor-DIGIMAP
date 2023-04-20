# DIGIMAP - Group 14
# Deculawan, Ryan Jay
# Tan, Lance Griffin
# Uy, Samuel Jedidiah
# Yu, Ethan Angelo

import streamlit as st
import unimodular_chaos_encryption as uce
from PIL import Image
from io import BytesIO
import numpy as np


def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def run(col1, col2, image_path, option):
    image = Image.open(image_path)
    image_array = np.array(image)
    col1.write("### Original Image :camera:")
    col1.image(image)

    pass_1_choices = uce.factors(
        image_array.size
    )  # Array of keys for pass 1 - need to reset per upload
    pass_1 = int(st.selectbox("Choose a key:", pass_1_choices))
    pass_2 = st.text_input("Enter your password", type="password")

    if pass_1 != "" and pass_2 != "":
        pass_2 = float("0." + pass_2 + "1")

        encrypted = uce.encrypt(image_path, pass_1, pass_2)
        image_array = np.array(encrypted)

        if option == "Encrypt":  # Encrypt
            col2.write("### Encrypted Image :closed_lock_with_key:")
            col2.image(encrypted)
            downloadable = convert_image(encrypted)
            st.sidebar.markdown("\n")
            st.sidebar.download_button(
                "Download encrypted image", downloadable, "encrypted.png", "image/png"
            )
        else:
            decrypted = uce.decrypt(image, pass_1, pass_2)
            col2.write("### Decrypted Image :unlock:")
            col2.image(decrypted)
            downloadable = convert_image(decrypted)
            st.sidebar.markdown("\n")
            st.sidebar.download_button(
                "Download decrypted image", downloadable, "decrypted.png", "image/png"
            )


def main():
    st.set_page_config(layout="wide", page_title="")
    st.write("## Image Encryptor/Decryptor")

    st.sidebar.write("## Upload and download :gear:")
    col1, col2 = st.columns(2)
    image_path = st.sidebar.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    st.sidebar.write(
        "This webapp employs unimodular chaos encryption to secure user-uploaded images. This state-of-the-art technique uses chaos theory and unimodular matrices to ensure data confidentiality and integrity, providing users with protection from unauthorized access or tampering. However, no encryption method can guarantee absolute security, as effectiveness depends on various factors such as encryption key strength, random number generation quality, and algorithm implementation. While this webapp uses encryption best practices, users should exercise caution and best practices when sharing sensitive data online."
    )
    if image_path is not None:
        option = st.select_slider(
            "What do you want to do?", options=["Encrypt", "Decrypt"]
        )
        run(col1, col2, image_path, option)
    st.write(
        """Welcome to our :lock: Image Encryptor/Decryptor! This tool allows you to protect your images from prying :eyes: by encrypting them with a password of your choice.

To use the tool, start by uploading the image :camera: you wish to encrypt or decrypt. Then, select a password :closed_lock_with_key: from the drop-down menu or enter your own password. Please make sure to remember the password you choose, as you will need it to decrypt the image later.

Next, use the toggle switch :gear: to select whether you want to encrypt or decrypt the image. If you choose to encrypt the image, the tool will generate a new, encrypted version of the image that can only be viewed by someone who knows the password. If you choose to decrypt the image, the tool will use the password you entered to reveal the original image.

Finally, you have the option to download the encrypted or decrypted image for safekeeping :inbox_tray:.

Thank you for using our :lock: Image Encryptor/Decryptor to keep your images secure!"""
    )


if __name__ == "__main__":
    main()
