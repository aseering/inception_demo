import os
import subprocess

from classify import run_inference_on_image
from classify_helpers import MODEL_DIR

import pytest


@pytest.fixture
def image_name():
    return os.path.join(MODEL_DIR, 'cropped_panda.jpg')


@pytest.fixture
def animal_name():
    return 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca'


def test_inference(image_name, animal_name):
    # Ignore score in case it varies slightly on different platforms.
    # Just check the name.
    human_readable_name, _ = run_inference_on_image(image_name)
    assert human_readable_name == animal_name


def test_inference_cmdline_default(animal_name):
    out = subprocess.check_output(['python3', 'classify.py'])
    assert animal_name.encode('utf-8') in out


def test_inference_cmdline_path(image_name, animal_name):
    out = subprocess.check_output(['python3', 'classify.py', image_name])
    assert animal_name.encode('utf-8') in out
