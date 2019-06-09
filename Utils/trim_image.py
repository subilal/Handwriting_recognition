import cv2

'''
This function will trim the blank rows/columns of an image
'''
def trim_image (image, hist_type = -1, trim_margin = 10):
    histogram = cv2.reduce(image, hist_type, cv2.REDUCE_AVG)
    histogram = histogram.reshape(-1)
    empty_rowcol_count = 0
    for rowcol in histogram:
        if (rowcol == 255):
            empty_rowcol_count = empty_rowcol_count + 1
    print (empty_rowcol_count)

