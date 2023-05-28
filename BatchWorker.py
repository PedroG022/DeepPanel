import os
import os.path

import DeepPanelRun

input_dir = "./MangaInput/"
output_dir = "./MangaOutput/"

folders = []

supported_extensions = ['jpg', 'jpeg', 'png']

for entry in os.listdir(input_dir):
    # entry = os.path.join(input_dir, entry)

    if os.path.isdir(os.path.join(input_dir, entry)):
        folders.append(entry)

for folder in folders:
    target_input = os.path.join(input_dir, folder) + "/"
    target_output = os.path.join(output_dir, folder) + "/"

    print(target_input)

    DeepPanelRun.processFolder(model=0, input_folder=target_input, panels_output_folder=target_output)

    # for entry in os.listdir(folder):
    #     entry = os.path.join(folder, entry)

    #     extension = entry[entry.index('.', len(entry) - 4) + 1:]
        
    #     if extension not in supported_extensions:
    #         print(f"Encountered a non-supported image extension {entry}!")

    #     print(extension)