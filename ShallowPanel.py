from ConnectedComponents import ConnectedComponentResult, find_components_x, DeepPanelResult, Panel
import numpy as np
import random
from random import randrange
import sys

def remove_small_areas_and_recover_border(connected_component_result: ConnectedComponentResult, width: int, height: int) -> ConnectedComponentResult:
    new_total_clusters = connected_component_result.total_clusters
    clusters_matrix = connected_component_result.clusters_matrix
    pixels_per_labels = connected_component_result.pixels_per_labels

    image_size = width * height

    label_removed = [ False for _ in range(1000) ]
    min_allowed_area = image_size * 0.03

    for i in range(width):
        for j in range(height):
            label = clusters_matrix[i][j]

            if label != 0:
                pixelsPerLabel = pixels_per_labels[label]

                if pixelsPerLabel < min_allowed_area:
                    clusters_matrix[i][j] = 0

                    if not label_removed[label]:
                        new_total_clusters -= 1
                        label_removed[label] = True


    result = ConnectedComponentResult(total_clusters=new_total_clusters, clusters_matrix=clusters_matrix, pixels_per_labels=pixels_per_labels)
    return result

def apply_scale_and_add_border(position: int, scale: float, border: int) -> int:
    float_position = float(position)
    scaled_position = float_position * scale
    return int(scaled_position) + border

def clamp(value: int, min: int, max: int) -> int:
    if value < min:
        return min
    elif value > max:
        return max
    else: 
        return value

def compute_border_size(original_image_width: int, original_image_height: int) -> int:
    if original_image_height > original_image_width:
        return original_image_width * 30 / 3056
    else:
        return original_image_height * 30 / 1988

def extract_panels_data(connected_components_result: ConnectedComponentResult, width: int, height: int, scale: float, original_image_width: int, original_image_height: int) -> DeepPanelResult:
    number_of_panels = connected_components_result.total_clusters
    current_normalized_label = 0

    print(f"Width: {width}")
    print(f"Height: {height}")

    normalized_labels = [0] * (width * height)

    min_x_values = [0] * (number_of_panels + 1)
    max_x_values = [0] * (number_of_panels + 1)
    min_y_values = [0] * (number_of_panels + 1)
    max_y_values = [0] * (number_of_panels + 1)

    for i in range(number_of_panels + 1):
        min_x_values[i] = float('inf')
        max_x_values[i] = float('-inf')
        min_y_values[i] = float('inf')
        max_y_values[i] = float('-inf')

    cluster_matrix = connected_components_result.clusters_matrix

    for i in range(width):
        for j in range(height):
            raw_label = cluster_matrix[i][j]

            if raw_label != 0:
                if normalized_labels[raw_label] == 0:
                    current_normalized_label += 1
                    normalized_labels[raw_label] = current_normalized_label

                normalized_label = normalized_labels[raw_label]

                min_x_values[normalized_label] = min(min_x_values[normalized_label], i)
                max_x_values[normalized_label] = max(max_x_values[normalized_label], i)
                min_y_values[normalized_label] = min(min_y_values[normalized_label], j)
                max_y_values[normalized_label] = max(max_y_values[normalized_label], j)

    panels = []

    horizontal_correction = 0
    vertical_correction = 0

    if  original_image_width < original_image_height:
        horizontal_correction = ((width * scale) - original_image_width) / 2
    else:
        vertical_correction = ((height * scale) - original_image_height) / 2
    
    border = compute_border_size(original_image_width, original_image_height)

    for i in range(number_of_panels):
        i += 1
        proposed_left = apply_scale_and_add_border(min_x_values[i], scale, -border) - horizontal_correction
        proposed_top = apply_scale_and_add_border(min_y_values[i], scale, -border) - vertical_correction
        proposed_right = apply_scale_and_add_border(max_x_values[i], scale, border) - horizontal_correction    
        proposed_bottom = apply_scale_and_add_border(max_y_values[i], scale, border) - vertical_correction

        panel = Panel()
        panel.left = clamp(proposed_left, 0, original_image_width)
        panel.top = clamp(proposed_top, 0, original_image_height)
        panel.right = clamp(proposed_right, 0, original_image_width)
        panel.bottom = clamp(proposed_bottom, 0, original_image_height)
        panels.append(panel)

    
    deep_panel_result = DeepPanelResult()
    deep_panel_result.connected_components = connected_components_result
    deep_panel_result.panels = panels

    return deep_panel_result

def extract_panels_info(labeled_matrix, width: int, height: int, scale: float, original_image_width: int, original_image_height: int) -> DeepPanelResult:
    improved_areas_result = find_components_x(labeled_matrix)
    improved_areas_result = remove_small_areas_and_recover_border(improved_areas_result, width, height)
    return extract_panels_data(improved_areas_result, width, height, scale, original_image_width,
                               original_image_height)

def parse_matrix(matrix):
    width = len(matrix[0])
    height = len(matrix[0][0])

    # print(f"Width: {width}")
    # print(f"Height: {height}")

    nm = [[0 for _ in range(width)] for _ in range(height)]

    for x in range(width):
        for y in range(height):
            val = -1
            background, border, content = matrix[0][x][y]

            if background >= content and background > border:
                val = 0
            elif border >= background and border >= content:
                val = 0
            else:
                val = 1

            nm[x][y] = val

    # label = find_components_x(nm)
    # print(label.total_clusters)

    return nm

def extract_panels(matrix, original_width, original_height):
    width = len(matrix[0])
    height = len(matrix[0][0])

    scale = max(original_height, original_width) / (width * height)

    # print(width)
    # print(height)

    return extract_panels_info(parse_matrix(matrix), width=width, height=height, scale=scale, original_image_height=original_height, original_image_width=original_width)