"""
This implementation of the Unimodular Encryption using Logistic Maps algorithm 
is adapted from the code written by Indra Bayu Muktyas, Sulistiawati, Samsul Arifin and 
published on his GitHub repository in 2020. 
The original code can be found at https://github.com/muktyas/encryption-unimodular-logistic-map.
"""

import numpy as np
from PIL import Image
import time
from functools import reduce
import io


# OBE
def elementary_row_operation(m, row_i, row_j, r):
    return m[row_i] + r * m[row_j]


def swap(m, row_i, row_j):
    m[row_i], m[row_j] = m[row_j], m[row_i]


"""
    This function generates a pseudo-random sequence of numbers using the logistic map equation with a given initial 
    value and a number of iterations.
    
    :param x0: float - initial value for the logistic map equation
    :param n: int - number of iterations for the logistic map equation
    :return: ndarray - sequence of n pseudo-random numbers between 0 and 255
    
    The logistic map equation is defined as follows:
    
    x_{n+1} = r * x_n * (1 - x_n)
    
    where x_n is the value at iteration n, x_{n+1} is the value at iteration n+1, and r is a constant between 3.7 and 4.
    In this implementation, r is fixed to 3.9.
    
    The function starts by iterating the logistic map equation for 1000 steps to ensure the sequence is not affected 
    by the initial value. Then, it computes the sequence of pseudo-random numbers by iterating the logistic map 
    equation n times and taking the integer part of the result multiplied by 1000 modulo 256.
"""


def log_map(x0, n):
    x = x0
    for i in range(1000):
        x = 3.9 * x * (1 - x)

    seq = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        x = 3.9 * x * (1 - x)
        seq[i] = x * 1000 % 256
    return seq


"""
    Generates a key matrix and its inverse for encryption and decryption purposes.
    Uses a logistic map to generate a sequence of numbers which is then used to fill an upper triangular matrix.
    This matrix is then augmented with an identity matrix and transformed into an echelon form using OBE to obtain its inverse matrix.

    Args:
        n (int): Size of the matrix.
        x0 (float): Initial value of the logistic map.

    Returns:
        tuple: A tuple containing two matrices - the key matrix and its inverse matrix.
"""


def generate_key(n, x0):
    count = n * (n - 1) // 2
    seq = log_map(x0, count + n - 1)

    m = np.eye(n)

    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = seq[idx]
            idx += 1

    for i in range(1, n):
        m[i] = elementary_row_operation(m, i, 0, seq[idx]) % 256
        idx += 1

    augmented = np.zeros((n, 2 * n))
    augmented[:, :n] = m
    augmented[:, n:] = np.eye(n)

    for col in range(n):
        for row in range(col + 1, n):
            augmented[row] = (
                elementary_row_operation(augmented, row, col, -augmented[row, col])
                % 256
            )

    for col in range(1, n):
        for row in range(col):
            augmented[row] = (
                elementary_row_operation(augmented, row, col, -augmented[row, col])
                % 256
            )

    inverse = augmented[:, n:]

    return m, inverse


"""
    Encrypts an image using a given key.
    The function takes the path to an image file 'image_path', along with the size 
    'size' and the initial value 'x0' used to generate the encryption key. It then 
    loads the image file and converts it to a NumPy array, generates the encryption 
    key using the 'generate_key()' function, reshapes the image array, multiplies 
    the key and the image array, reshapes the resulting cipher, and converts it to 
    a Pillow Image object. The encrypted image is returned.

    Args:
    - image_path (str): The path to the image file.
    - size (int): The size of the image.
    - x0 (int): The initial value used to generate the encryption key.

    Returns:
    - PIL.Image.Image: The encrypted image as a Pillow Image object.
"""


def encrypt(image_path, size, x0):
    import time
    import numpy as np
    from PIL import Image

    start_time = time.time()

    # Load the image
    image = Image.open(image_path)
    image_array = np.array(image)

    # Generate the key
    key, inverse_key = generate_key(size, x0)

    # Reshape the image
    image_reshaped = image_array.reshape((size, image_array.size // size))

    # Multiply the key and the image
    multiplied = np.dot(key, image_reshaped) % 256

    # Reshape the cipher
    cipher_array = multiplied.reshape(image_array.shape).astype(np.uint8)

    # Convert the cipher array to a Pillow Image object and return it
    cipher_image = Image.fromarray(cipher_array)
    return cipher_image


"""
    Decrypts an image using a given key.

    The function takes a grayscale image 'gb' as input, along with its size 'size' 
    and the initial value 'x0' used to generate the encryption key. It then reshapes 
    the image array and applies the inverse of the encryption key to decrypt the image. 
    The resulting decrypted image is converted to a Pillow Image object and returned.
    
    Args:
    - gb (numpy.ndarray): A grayscale image represented as a NumPy array.
    - size (int): The size of the image.
    - x0 (int): The initial value used to generate the encryption key.
    
    Returns:
    - PIL.Image.Image: The decrypted image as a Pillow Image object.
"""


def decrypt(gb, size, x0):
    import numpy as np
    from PIL import Image

    print("Decryption process begin.")

    # Convert the grayscale image to a NumPy array
    image_array = np.array(gb)

    # Generate the decryption key using the 'generate_key()' function
    key, inverse = generate_key(size, x0)

    # Reshape the image array
    image_array_reshaped = image_array.reshape((size, image_array.size // size))

    # Apply the inverse of the encryption key to decrypt the image
    decrypted_image_array = np.dot(inverse, image_array_reshaped) % 256
    decrypted_image_array_reshaped = decrypted_image_array.reshape(
        image_array.shape
    ).astype(np.uint8)

    # Convert the decrypted image array to a Pillow Image object and return it
    decrypted_image = Image.fromarray(decrypted_image_array_reshaped)
    return decrypted_image


"""
    Compares two image files to determine if they are the same.
    The function takes the file paths of two image files, 'original_image' and 'decrypted_image', 
    opens them with the Pillow Image module, and converts them to NumPy arrays. 
    It then uses the 'all()' method of NumPy arrays to compare the two images, 
    returning "Both images are the same." if they are identical, and "Both images are different." otherwise.
    
    Args:
    - original_image (str): The file path of the original image.
    - decrypted_image (str): The file path of the decrypted image to be compared.
    
    Returns:
    - str: A string indicating whether the images are the same or different.
"""


def check_same(original_image, decrypted_image):
    from PIL import Image
    import numpy as np

    # Open the original and decrypted images with the Pillow Image module
    image_a = Image.open(original_image)
    image_b = Image.open(decrypted_image)

    # Convert the images to NumPy arrays
    image_a_array = np.array(image_a)
    image_b_array = np.array(image_b)

    # Compare the two arrays using the 'all()' method
    if (image_a_array == image_b_array).all():
        return "Both images are the same."
    else:
        return "Both images are different."


"""
    Returns a list of all factors of the input integer 'n' up to a maximum of 1000.
    The function uses a set comprehension to generate all factors of 'n' in a raw format, 
    then sorts the factors in ascending order and filters out any factors greater than 1000.

    Args:
    - n (int): The integer to find factors of.

    Returns:
    - List[int]: A list of all factors of 'n' up to a maximum of 1000, sorted in ascending order.
"""


def factors(n):
    from functools import reduce

    # Use set comprehension to generate all factors of 'n' in a raw format
    raw_factors = set(
        reduce(
            list.__add__,
            ([i, n // i] for i in range(2, int(n**0.5) + 1) if n % i == 0),
        )
    )

    # Sort the factors in ascending order and filter out any factors greater than 1000
    sorted_factors = sorted(raw_factors)
    return [i for i in sorted_factors if i < 1000]


if __name__ == "__main__":
    pass
