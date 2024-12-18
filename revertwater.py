import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from PIL import Image
import scipy.io
import cv2

# Enhance watermark function
def enhance_watermark(recovered_watermark):
    # Ensure input is single-channel and scaled to [0, 255]
    img = np.clip(recovered_watermark, 0, 1)  # Clip values to [0, 1]
    img = (img * 255).astype(np.uint8)        # Convert to 8-bit grayscale

    if len(img.shape) == 3:  # Convert multi-channel to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply sharpening filter
    img_blur = cv2.GaussianBlur(img, (5, 5), 2)
    img_sharp = cv2.addWeighted(img, 2.0, img_blur, -1.0, 0)

    # Adaptive binarization
    img_bin = cv2.adaptiveThreshold(img_sharp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return img_bin


# Extract watermark function
def extract_watermark(original_image, watermarked_image, M, N, alpha, mark_shape):
    # Perform 2D Fourier Transform on original and watermarked images
    FA = fft2(original_image)
    FB = fft2(watermarked_image)
    
    # Extract frequency domain watermark
    G = (FB - FA) / alpha
    G = np.real(G)

    G = G[:, :, 0]

    # G = enhance_watermark(G)

    # Recover watermark from extracted frequencies
    watermark_extracted = np.zeros(mark_shape)
    for i in range(mark_shape[0]):
        for j in range(mark_shape[1]):
            # watermark_extracted[M[i] % mark_shape[0], N[j] % mark_shape[1]] = G[i, j]
            orig_i = np.where(M == i)[0][0]  # 找到原始位置
            orig_j = np.where(N == j)[0][0]
            # print(orig_i, orig_j)
            watermark_extracted[i, j] = G[orig_i, orig_j]
            # watermark_extracted[orig_i % mark_shape[0], orig_j % mark_shape[1]] = G[i, j]
    return watermark_extracted

# Load images and normalize
im = np.array(Image.open("chuying.png")).astype(np.float64) / 255.0
watermarked_image = np.array(Image.open("watermarked_image.png")).astype(np.float64) / 255.0
mark = np.array(Image.open("luna.png")).astype(np.float64) / 255.0

# Load encoding data
endcode_data = scipy.io.loadmat('endcode.mat')
M = endcode_data['M'][0]  # Row indices
N = endcode_data['N'][0]  # Column indices
alpha = endcode_data['alpha'][0][0]

# Extract watermark
extracted_watermark = extract_watermark(im, watermarked_image, M, N, alpha, mark.shape)

# Display original watermark
plt.figure()
plt.imshow(mark, cmap='gray')
plt.title('Original Watermark')
plt.show()

# Display extracted watermark
plt.figure()
plt.imshow(extracted_watermark, cmap='gray')
# plt.imshow(extracted_watermark, cmap='gray')
plt.title('Extracted Watermark')
plt.show()

# Save extracted watermark
Image.fromarray((extracted_watermark * 255).astype(np.uint8)).save('extracted_watermark.png')
