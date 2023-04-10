import os
from jinja2 import Environment, FileSystemLoader
from utils import files_in_folder

OUTPUT_FOLDER = "./output/"
TEST_FOLDER = "./test_input/"
MASK_FOLDER = "./output/masks/"
REPORTS_PATH = "./reports"

class PredictionResult:

    def __init__(self, page, pred, mask):
        self.page = page
        self.pred = pred
        self.mask = mask

        self.output_folder = OUTPUT_FOLDER
        self.test_folder = TEST_FOLDER
        self.mask_folder = MASK_FOLDER

def generate_output_template():
    output_images = files_in_folder(OUTPUT_FOLDER)
    test_images = files_in_folder(TEST_FOLDER)
    true_masks_images = files_in_folder(MASK_FOLDER)

    template_predictions = list()
    
    for i in range(len(output_images)):
        template_predictions.append(PredictionResult(test_images[i], output_images[i], true_masks_images[i]))

    loader = FileSystemLoader("./templates")
    env = Environment(loader=loader)

    template = env.get_template('index.html')
    template_output = template.render(predictions=template_predictions)

    if not os.path.exists(REPORTS_PATH):
        os.makedirs(REPORTS_PATH)

    text_file = open(f"{REPORTS_PATH}/index.html", "w")
    text_file.write(template_output)
    text_file.close()

if __name__ == "__main__":
    generate_output_template()