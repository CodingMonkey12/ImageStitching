import cv2 as cv


def remove_black_edge(image):
    # median filter, to remove the noise interference that may be contained in the black edge
    img = cv.medianBlur(image, 5)
    # adjust crop effect
    b = cv.threshold(img, 15, 255, cv.THRESH_BINARY)
    binary_image = b[1]
    binary_image = cv.cvtColor(binary_image, cv.COLOR_BGR2GRAY)

    x = binary_image.shape[0]
    y = binary_image.shape[1]
    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(y):
            if binary_image[i][j] == 255:
                edges_x.append(i)
                edges_y.append(j)

    left = min(edges_x)
    right = max(edges_x)
    width = right - left
    bottom = min(edges_y)
    top = max(edges_y)
    height = top - bottom

    pre1_picture = image[left:left + width, bottom:bottom + height]
    return pre1_picture


if __name__ == '__main__':
    image = cv.imread('images/output/panorama.jpg')
    new_image = remove_black_edge(image)
    cv.imwrite('images/output/' + 'new_panorama.jpg', new_image)
