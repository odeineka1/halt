"""Tests for detect_person(). Requires yolo-Weights/yolov8n.pt to be present
(same requirement as running main.py), since importing main.py loads the model.
"""

from main import detect_person


def test_detect_person_left_side():
    # Box fully in the left half of a 1280px-wide frame -> Player 1
    assert detect_person(0, 0, 100, 100, frame_width=1280) == 1


def test_detect_person_right_side():
    # Box fully in the right half -> Player 2
    assert detect_person(700, 0, 900, 100, frame_width=1280) == 2


def test_detect_person_straddling_midline_left_majority():
    # Box crosses the midline (640) but sits mostly on the left -> Player 1
    assert detect_person(500, 0, 700, 100, frame_width=1280) == 1


def test_detect_person_straddling_midline_right_majority():
    # Box crosses the midline but sits mostly on the right -> Player 2
    assert detect_person(600, 0, 900, 100, frame_width=1280) == 2