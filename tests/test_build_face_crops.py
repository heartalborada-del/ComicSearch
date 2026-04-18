import unittest

import numpy as np

from scripts.build_face_crops import detect_faces


class FakeBox:
    def __init__(self, coords, score):
        self.xyxy = np.array([coords], dtype=np.float32)
        self.conf = FakeTensorValue(score)


class FakeTensorValue:
    def __init__(self, value):
        self._value = float(value)

    def numel(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self

    def item(self):
        return self._value


class FakeBoxes:
    def __init__(self, boxes):
        self._boxes = boxes

    def __iter__(self):
        return iter(self._boxes)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = FakeBoxes(boxes)


class FakeDetector:
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        device = kwargs.get("device")
        if device and device != "cpu":
            raise NotImplementedError("Could not run 'torchvision::nms' with arguments from the 'CUDA' backend.")
        return [FakeResult([FakeBox([1, 2, 10, 12], 0.9)])]


class DetectFacesTests(unittest.TestCase):
    def test_detect_faces_retries_on_cpu_when_cuda_nms_is_unavailable(self):
        detector = FakeDetector()
        image = np.zeros((20, 20, 3), dtype=np.uint8)

        detections = detect_faces(
            image=image,
            detector=detector,
            confidence_threshold=0.35,
            max_detections_per_image=6,
            class_ids=[0],
            device="cuda:0",
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].bbox, (1, 2, 10, 12))
        self.assertEqual(len(detector.calls), 2)
        self.assertEqual(detector.calls[0]["device"], "cuda:0")
        self.assertEqual(detector.calls[1]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()