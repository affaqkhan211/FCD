# import cv2
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import base64


# # def preprocess_image(base64_image):
# #     # Decode base64 image
# #     img_data = base64.b64decode(base64_image)
# #     nparr = np.frombuffer(img_data, np.uint8)
# #     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# #     # Apply your preprocessing steps
# #     for i in range(len(img[:, 0, 0])):
# #         for j in range(len(img[0, :, 0])):
# #             R = int(img[i, j, 0])
# #             G = int(img[i, j, 1])
# #             B = int(img[i, j, 2])

# #             sum_col = R + G + B

# #             if (sum_col > 180) & (R > 200) & (G > 200) & (B > 200):
# #                 img[i, j, 0] = img[i - 1, j - 1, 0]
# #                 img[i, j, 1] = img[i - 1, j - 1, 1]
# #                 img[i, j, 2] = img[i - 1, j - 1, 2]

# #     # Save the processed image locally
# #     preprocess_folder = '../AI_Detect_Fake_Currency/src/screens/processedImage'  # Change this to your desired folder
# #     if not os.path.exists(preprocess_folder):
# #         os.makedirs(preprocess_folder)
# #     processed_image_path = os.path.join(preprocess_folder, 'processed_image')
    
# #     cv2.imwrite(processed_image_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# #     # Print the path to the console
# #     print(f"Processed image saved at: {processed_image_path}")

# #     # Convert the processed image to binary data
# #     _, img_encoded = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# #     processed_image_binary = img_encoded.tobytes()

# #     # Convert binary data to base64
# #     processed_image_base64 = base64.b64encode(processed_image_binary).decode('utf-8')

# #     # Return the binary data and base64-encoded image
# #     return processed_image_binary, processed_image_base64


# def preprocess_image(image_path):
#     # Read the image
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Apply your preprocessing steps
#     for i in range(len(img[:, 0, 0])):
#         for j in range(len(img[0, :, 0])):
#             R = int(img[i, j, 0])
#             G = int(img[i, j, 1])
#             B = int(img[i, j, 2])

#             sum_col = R + G + B

#             if (sum_col > 180) & (R > 200) & (G > 200) & (B > 200):
#                 img[i, j, 0] = img[i - 1, j - 1, 0]
#                 img[i, j, 1] = img[i - 1, j - 1, 1]
#                 img[i, j, 2] = img[i - 1, j - 1, 2]

#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # red color boundaries [B, G, R]
#     lower = [np.mean(image[:, :, i] - np.std(image[:, :, i]) / 3) for i in range(3)]
#     upper = [250, 250, 250]

#     # create NumPy arrays from the boundaries
#     lower = np.array(lower, dtype="uint8")
#     upper = np.array(upper, dtype="uint8")

#     # find the colors within the specified boundaries and apply
#     mask = cv2.inRange(image, lower, upper)
#     output = cv2.bitwise_and(image, image, mask=mask)

#     ret, thresh = cv2.threshold(mask, 40, 255, 0)

#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#     if len(contours) != 0:
#         # find the biggest contour (c) by the area
#         c = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(c)

#         # draw the biggest contour (c) in green
#         cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 5)

#     # Crop the image using the coordinates of the bounding rectangle
#     cropped_image = image[y:y + h, x:x + w]

#     # Save the processed image in the "preprocess" folder with a fixed filename
#     preprocess_folder = '../AI_Detect_Fake_Currency/src/screens/processedImage'
#     if not os.path.exists(preprocess_folder):
#         os.makedirs(preprocess_folder)

#     processed_image_path = os.path.join(preprocess_folder, 'processed_image.jpg')
#     cv2.imwrite(processed_image_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

#     # Display the processed image (optional)
#     plt.imshow(cropped_image, cmap='gray')
#     plt.title("Processed Image")
#     plt.show()

#     # Convert the processed image to binary data
#     _, img_encoded = cv2.imencode('.png', cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
#     processed_image_binary = img_encoded.tobytes()

#     # Return the binary data of the processed image
#     return processed_image_binary, processed_image_path






















import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def is_salt_and_pepper_noise(image, threshold=0.5, kernel_size=3):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply morphological opening to identify isolated pixels
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # Calculate the absolute difference between the original and opened images
    diff = cv2.absdiff(gray, opened)

    # Calculate the percentage of non-zero pixels in the difference image
    non_zero_percentage = np.count_nonzero(diff) / np.prod(image.shape[0:2])

    ##### Just for visualization and debug; remove in the final code
    plt.imshow(diff, cmap='gray')
    plt.colorbar()
    plt.show()
    ##### Just for visualization and debug; remove in the final code

    # Percentage threshold; above: potential salt and pepper noise
    return non_zero_percentage > threshold


# def is_valid(image):
#     # Your noise detection code here
#     # ...
#     # Convert image to HSV color space
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Calculate histogram of saturation channel
#     s = cv2.calcHist([image], [1], None, [256], [0, 256])

#     # Calculate percentage of pixels with saturation >= p
#     p = 0.05
#     s_perc = np.sum(s[int(p * 255):-1]) / np.prod(image.shape[0:2])

#     ##### Just for visualization and debug; remove in final
#     plt.plot(s)
#     plt.plot([p * 255, p * 255], [0, np.max(s)], 'r')
#     plt.text(p * 255 + 5, 0.9 * np.max(s), str(s_perc))
#     plt.show()
#     ##### Just for visualization and debug; remove in final

#     # Percentage threshold; above: valid image, below: noise
#     s_thr = 0.5
# #D:/FYP(Dataset)/archive/data-rescaled/1000_back/IMG20211228225540.jpg
#     return s_perc > s_thr



def determine_noise_type(image):
    # Calculate the mean and standard deviation of pixel values
    mean_pixel_value = np.mean(image)
    std_pixel_value = np.std(image)

    # Define thresholds for determining noise type
    salt_and_pepper_threshold = 150  # Adjust as needed
    spackle_threshold = 10  # Adjust as needed

    # Check if the image has salt-and-pepper noise
    if std_pixel_value > salt_and_pepper_threshold:
        return "salt_and_pepper"

    # Check if the image has spackle noise
    elif mean_pixel_value < spackle_threshold:
        return "spackle"

    # If neither type is detected, return "unknown"
    else:
        return "unknown"



def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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


    

    # Check if the cropped image is valid (does not contain noise)
    if is_salt_and_pepper_noise(cropped_image):
     
        # Save the processed image in the "preprocess" folder with a fixed filename
        denoised_image = cv2.medianBlur(cropped_image, 5)  # Adjust the kernel size as needed

        # Save the processed image in the "preprocess" folder with a fixed filename
        preprocess_folder = '../AI_Detect_Fake_Currency/src/screens/processedImage'
        if not os.path.exists(preprocess_folder):
            os.makedirs(preprocess_folder)

        processed_image_path = os.path.join(preprocess_folder, 'processed_image.jpg')
        cv2.imwrite(processed_image_path, cv2.cvtColor(denoised_image, cv2.COLOR_RGB2BGR))

        # Display the processed image (optional)
        plt.imshow(denoised_image, cmap='gray')
        plt.title("Processed Image")
        plt.show()

        # Convert the processed image to binary data
        _, img_encoded = cv2.imencode('.png', cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))
        processed_image_binary = img_encoded.tobytes()

        # Return the binary data of the processed image
        return processed_image_binary, processed_image_path
    else:
   
        preprocess_folder = '../AI_Detect_Fake_Currency/src/screens/processedImage'
        if not os.path.exists(preprocess_folder):
            os.makedirs(preprocess_folder)

        processed_image_path = os.path.join(preprocess_folder, 'processed_image.jpg')
        cv2.imwrite(processed_image_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

        # Display the processed image (optional)
        plt.imshow(cropped_image, cmap='gray')
        plt.title("Processed Image")
        plt.show()

        # Convert the processed image to binary data
        _, img_encoded = cv2.imencode('.png', cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        processed_image_binary = img_encoded.tobytes()

        # Return the binary data of the processed image
        return processed_image_binary, processed_image_path

# Example usage:

