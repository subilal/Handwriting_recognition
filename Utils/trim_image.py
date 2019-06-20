import cv2

'''
This function will trim the blank rows/columns of an image
'''

def trim_image_with_component(image, padding=10):
    nr_rows = image.shape[0]
    nr_cols = image.shape[1]

    rows_reduced = cv2.reduce(image, 1, cv2.REDUCE_MIN).reshape(-1)
    cols_reduced = cv2.reduce(image, 0, cv2.REDUCE_MIN).reshape(-1)

    # First, determine the number of blanc lines from the top
    threshold = 250
    blancs_top = count_blancs_until_component(rows_reduced, min_threshold=threshold)
    new_top = max (blancs_top - padding, 0)

    blancs_bottom = count_blancs_until_component(rows_reduced[::-1], min_threshold=threshold)
    new_bottom = min (nr_rows - blancs_bottom + padding, nr_rows)

    blancs_left = count_blancs_until_component(cols_reduced, min_threshold=threshold)
    new_left = max (blancs_left - padding, 0)

    blancs_right = count_blancs_until_component(cols_reduced[::-1], min_threshold=threshold)
    new_right = min (nr_cols - blancs_right + padding, nr_cols)

    # Crop image
    trimmed_image = image[new_top:new_bottom, new_left:new_right]

    return trimmed_image


def count_blancs_until_component(line, min_threshold=255):
    blancs = 0
    for pixel in line:
        if pixel < min_threshold:
            break
        blancs = blancs + 1

    return blancs

