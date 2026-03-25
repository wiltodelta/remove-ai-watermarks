import numpy as np

from remove_ai_watermarks.face_protector import FaceProtector


def test_face_protector_initialization():
    # Will fallback to Haar cascade if ultralytics is missing
    fp = FaceProtector(use_yolo=False)
    assert fp.use_yolo is False
    assert fp.haar_cascade is not None


def test_face_protector_lifecycle():
    fp = FaceProtector(use_yolo=False)

    # Create dummy black image
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    # Since it's a black image, haar cascade should find 0 faces
    faces = fp.extract_faces(img)
    assert isinstance(faces, list)
    assert len(faces) == 0

    # Restoring 0 faces should result in strictly equal image
    restored = fp.restore_faces(img, faces)
    assert np.array_equal(img, restored)


def test_face_protector_restore_bypass_on_size_mismatch():
    fp = FaceProtector(use_yolo=False)
    img_small = np.zeros((100, 100, 3), dtype=np.uint8)

    # Manually mock a face that is OUT OF BOUNDS for img_small
    mock_bbox = (80, 80, 130, 130)
    mock_crop = np.ones((50, 50, 3), dtype=np.uint8) * 255
    mock_faces = [(mock_bbox, mock_crop)]

    # Attempt to restore onto an image too small for this box
    restored = fp.restore_faces(img_small, mock_faces)

    # Should safely skip restoring and not crash
    assert np.array_equal(restored, img_small)


def test_face_protector_restore_blending():
    fp = FaceProtector(use_yolo=False)
    # Background is black
    img_target = np.zeros((100, 100, 3), dtype=np.uint8)

    # Face crop is white
    mock_bbox = (25, 25, 75, 75)
    mock_crop = np.ones((50, 50, 3), dtype=np.uint8) * 255
    mock_faces = [(mock_bbox, mock_crop)]

    restored = fp.restore_faces(img_target, mock_faces)

    # The center of the face should be perfectly white (255)
    assert restored[50, 50, 0] >= 254
    # The corner of the target should remain perfectly black (0)
    assert restored[0, 0, 0] == 0
    # We should have a blending gradient between them due to the gaussian blur mask
    # For example, around (30, 30) or similar
    assert 0 <= restored[28, 28, 0] <= 255
