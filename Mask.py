import numpy as np
import cv2

def generate_panels_bitmap(source_bitmap, panels):
    temp_bitmap = source_bitmap.copy()

    canvas = np.zeros_like(temp_bitmap)

    for panel in panels:
        color = color_for_label(panel.number_in_page + 2)

        cv2.rectangle(
            canvas, 
            (int(panel.left), int(panel.top)), 
            (int(panel.width), int(panel.height)), 
            color, 
            thickness=-1
        )

    result = cv2.addWeighted(temp_bitmap, 0.7, canvas, 0.3, 0)
    return result

def color_for_label(label):
    color_dict = {
        -1: (0, 0, 0), 
        0: (255, 0, 0), 
        1: (0, 0, 255), 
        2: (0, 255, 0), 
        3: (255, 255, 0), 
        4: (255, 0, 255), 
        5: (255, 0, 128), 
        6: (76, 0, 79), 
        7: (8, 69, 24), 
        8: (40, 139, 143), 
        9: (143, 121, 40), 
        10: (217, 147, 212), 
        11: (84, 27, 20), 
        12: (163, 86, 13)
    }
    return color_dict.get(label, (255, 255, 255))