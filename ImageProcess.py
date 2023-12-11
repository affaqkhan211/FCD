import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import base64

def is_valid(img):
    # Convert image to HSV color space
    #haha
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Calculate histogram of saturation channel
    s = cv2.calcHist([image], [1], None, [256], [0, 256])

    # Calculate percentage of pixels with saturation >= p
    p = 0.05
    s_perc = np.sum(s[int(p * 255):-1]) / np.prod(image.shape[0:2])

    # Just for visualization and debug; remove in the final version
    plt.plot(s)
    plt.plot([p * 255, p * 255], [0, np.max(s)], 'r')
    plt.text(p * 255 + 5, 0.9 * np.max(s), str(s_perc))
    plt.show()

    # Percentage threshold; above: valid image, below: noise
    s_thr = 0.5

    return s_perc > s_thr

def preprocess_image(base64_image):
    # Decode base64 image
    img_data = base64.b64decode(base64_image)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Apply your preprocessing steps
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

    # Save the processed image locally
    preprocess_folder = './result'  # Change this to your desired folder
    if not os.path.exists(preprocess_folder):
        os.makedirs(preprocess_folder)

    processed_image_path = os.path.join(preprocess_folder, 'processed_image.png')
    cv2.imwrite(processed_image_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Print the path to the console
    print(f"Processed image saved at: {processed_image_path}")

    # Convert the processed image to binary data
    _, img_encoded = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    processed_image_binary = img_encoded.tobytes()

    # Convert binary data to base64
    processed_image_base64 = base64.b64encode(processed_image_binary).decode('utf-8')

    # Return the binary data and base64-encoded image
    return processed_image_binary, processed_image_base64