import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify




font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 25,
        }

#D:/FYP(Dataset)/archive/data-rescaled/500_back/IMG20211228224708.jpg
# Specify the path of the image you want to process
image_path = "C:/Users/Nabeel Ahmad/Desktop/note.jpg"

# Crop the image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

for i in range(len(img[:, 0, 0])):
    for j in range(len(img[0, :, 0])):
        R = int(img[i, j, 0])
        G = int(img[i, j, 1])
        B = int(img[i, j, 2])

        sum_col = R + G + B

        if (sum_col > 180) & (R > 200) & (G > 200) & (B > 200):
            img[i, j, 0] = img[i - 1, j - 1, 0]
            img[i, j, 1] = img[i - 1, j - 1, 1]
            img[i, j, 2] = img[i - 1, j - 1, 2]

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# red color boundaries [B, G, R]
lower = [np.mean(image[:, :, i] - np.std(image[:, :, i]) / 3) for i in range(3)]
upper = [250, 250, 250]

# create NumPy arrays from the boundaries
lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")

# find the colors within the specified boundaries and apply
mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask=mask)

ret, thresh = cv2.threshold(mask, 40, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if len(contours) != 0:
    # find the biggest contour (c) by the area
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # draw the biggest contour (c) in green
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 5)

# Crop the image using the coordinates of the bounding rectangle
cropped_image = image[y:y + h, x:x + w]

# Convert the cropped image to grayscale
cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)

# Display the cropped image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(cropped_image), plt.title("Cropped Image", fontdict=font)
plt.show()
# Apply noise removal to the cropped image
min_val, max_val, _, _ = cv2.minMaxLoc(cropped_image_gray)

if min_val == 0 and max_val == 255:
    print("Image may contain salt-and-pepper noise.")
    img_filtered_median = cv2.medianBlur(cropped_image_gray, 5)
    kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_sharpened_median = cv2.filter2D(img_filtered_median, -1, kernel_sharpening)

    # Display the original, filtered, and sharpened images
    plt.subplot(1, 2, 2), plt.imshow(img_filtered_median, cmap='gray'), plt.title("Filtered Image (Median)", fontdict=font)
    plt.show()

# Similar logic can be applied for other noise removal techniques...
gray_var = np.var(img)
speckle_threshold = 100
if gray_var > speckle_threshold:
    print("Image may contain speckle noise.")

    # Remove speckle noise using a median filter
    img_filtered_speckle = cv2.medianBlur(img, 5)  # Adjust the kernel size as needed

    # Apply a sharpening filter to reduce blurriness
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    img_sharpened_speckle = cv2.filter2D(img_filtered_speckle, -1, kernel_sharpening)

    # Display the original, filtered, and sharpened images
    cv2.imshow('Original Image - Speckle', img)
    cv2.imshow('Filtered Image (Speckle)', img_filtered_speckle)

    key = cv2.waitKey(0)  # Wait for a key press
    if key == 27:  # 27 is the ASCII code for the Esc key
        cv2.destroyAllWindows()
    # Implement your speckle noise removal code here

# Example code to detect Gaussian noise using frequency domain analysis (FFT)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift) + 1)

# Check if the magnitude spectrum has a peak at the center (indicative of Gaussian noise)
gaussian_threshold = 100  # Adjust the threshold based on your image characteristics

if np.max(magnitude_spectrum) > gaussian_threshold:
    print("Image may contain Gaussian noise.")

    # Remove Gaussian noise using a Gaussian filter
    img_filtered_gaussian = cv2.GaussianBlur(img, (5, 5), 0)  # Adjust the kernel size as needed

    # Apply a sharpening filter to reduce blurriness
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    img_sharpened_gaussian = cv2.filter2D(img_filtered_gaussian, -1, kernel_sharpening)

    # Display the original, filtered, and sharpened images
    cv2.imshow('Original Image - Gaussian', img)
    cv2.imshow('Filtered Image (Gaussian)', img_filtered_gaussian)

    key = cv2.waitKey(0)  # Wait for a key press
    if key == 27:  # 27 is the ASCII code for the Esc key
        cv2.destroyAllWindows()













