import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from PIL import Image
import scipy.io
n = 10
perm = np.random.permutation(n)
print(perm)
# Load images and normalize
im = np.array(Image.open("chuying.png")).astype(np.float64) / 255.0
mark = np.array(Image.open("luna.png")).astype(np.float64) / 255.0

# Display the original image
# plt.figure()
# plt.imshow(im)
# plt.title('Original Image')
# plt.show()

# Get image and watermark sizes
imsize = im.shape
marksize = mark.shape
print(imsize)
print(marksize)

# Generate encoding positions (M, N) and alpha
M = np.random.permutation(imsize[0])
N = np.random.permutation(imsize[1])
alpha = 20  # Watermark strength

# Save M, N, and alpha to .mat file
scipy.io.savemat('endcode.mat', {'M': M, 'N': N, 'alpha': alpha})

# Embed watermark into the original image
mark_ = np.copy(im)
for i in range(imsize[0]):
    for j in range(imsize[1]):
        mark_[i, j, :] = mark[M[i] % marksize[0], N[j] % marksize[1]]

scipy.io.savemat('endcode.mat', {'M': M, 'N': N, 'alpha': alpha, "mark_": mark_})
# Perform 2D Fourier Transform on the original image
FA = fft2(im)
print(mark_[0,0,1])
FB = FA + alpha * mark_  # Add watermark in the frequency domain

# Perform Inverse Fourier Transform
FAO = ifft2(FB)
FAO_abs = np.real(FAO)  # Take the magnitude to visualize

# Clip values to ensure valid range
FAO_abs[FAO_abs > 1] = 1

# Display the watermarked image
plt.figure()
plt.imshow(FAO_abs, cmap='gray')
plt.title('Watermarked Image_')
plt.show()

# mark_ = FA
# mark_shape = marksize
# watermark_extracted = np.zeros(mark_shape)
# for i in range(mark_shape[0]):
#     for j in range(mark_shape[1]):
#         # watermark_extracted[M[i] % mark_shape[0], N[j] % mark_shape[1]] = G[i, j]
#         orig_i = np.where(M == i)[0][0]  # 找到原始位置
#         orig_j = np.where(N == j)[0][0]
#         # print(orig_i, orig_j)
#         # if orig_i < mark_shape[0] and orig_j < mark_shape[1]:
#         #     watermark_extracted[orig_i, orig_j] = np.abs(mark_[i, j, 0])
#         watermark_extracted[i, j] = mark_[orig_i, orig_j, 0]

# plt.figure()
# plt.imshow(watermark_extracted, cmap='gray')
# plt.title('watermark_extracted')
# plt.show()

plt.figure()
plt.imshow(mark, cmap='gray')
plt.title('watermark_extracted')
plt.show()

# Save the watermarked image
Image.fromarray((FAO_abs * 255).astype(np.uint8)).save('watermarked_image.png')
