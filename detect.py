import json
from pathlib import Path
from typing import Dict

import click
import cv2
import tensorflow as tf
from tqdm import tqdm
import numpy as np


def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        A dictionary with the number of each object.
    """
    model_path = './'
    model = tf.saved_model.load(model_path)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img2 = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, fixed_thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    img_thresh_inverted = cv2.bitwise_not(fixed_thresh)

    contours, _ = cv2.findContours(img_thresh_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    aspen = 0
    birch = 0
    hazel = 0
    maple = 0
    oak = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        leaf_roi = img[y-10:y+h+10, x-10:x+w+10]

        if h < 50 and w < 50:
            continue

        leaf_roi_resized = cv2.resize(leaf_roi, (224, 224))
        leaf_roi_float = leaf_roi_resized.astype(np.float32) / 255.0 
        leaf_roi_input = np.expand_dims(leaf_roi_float, axis=0)

        result = model(leaf_roi_input)
        class_idx = np.argmax(result, axis=1)[0]

        class_names = ['aspen', 'birch', 'hazel', 'maple', 'oak']
        leaf_name = class_names[class_idx]

        if leaf_name == 'aspen':
            aspen += 1
        elif leaf_name == 'birch':
            birch += 1
        elif leaf_name == 'hazel':
            hazel += 1
        elif leaf_name == 'maple':
            maple += 1
        elif leaf_name == 'oak':
            oak += 1
        
        cv2.rectangle(img2, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)

    return {'aspen': aspen, 'birch': birch, 'hazel': hazel, 'maple': maple, 'oak': oak}

@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        leaves = detect(str(img_path))
        results[img_path.name] = leaves

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
