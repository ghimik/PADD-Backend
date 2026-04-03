import asyncio
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


def install_fake_core_modules():
    package_names = [
        "core",
        "core.Neuretus_XElite",
        "core.Neuretus_XElite.core",
    ]

    for name in package_names:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module

    detectors = types.ModuleType("core.Neuretus_XElite.core.detectors")
    geometry = types.ModuleType("core.Neuretus_XElite.core.geometry")
    ocr = types.ModuleType("core.Neuretus_XElite.core.ocr")
    pdfyer = types.ModuleType("core.Neuretus_XElite.core.pdfyer")

    class Placeholder:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    detectors.MalboroDetector = Placeholder
    detectors.ComputantisDetector = Placeholder
    detectors.CornerBaneRefiner = Placeholder
    geometry.HomographyCorrector = Placeholder
    ocr.OCRProcessor = Placeholder
    ocr.RotationDetector = Placeholder
    ocr.DocumentEnhancer = Placeholder
    pdfyer.PDFEngine = Placeholder

    sys.modules[detectors.__name__] = detectors
    sys.modules[geometry.__name__] = geometry
    sys.modules[ocr.__name__] = ocr
    sys.modules[pdfyer.__name__] = pdfyer


def install_fake_python_multipart():
    python_multipart = types.ModuleType("python_multipart")
    python_multipart.__version__ = "0.0.20"
    sys.modules["python_multipart"] = python_multipart


WORKER_SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKER_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKER_SERVICE_ROOT))

install_fake_core_modules()
install_fake_python_multipart()

import app.main as main_module  # noqa: E402


class DummyUploadFile:
    def __init__(self, content: bytes):
        self.file = io.BytesIO(content)


class MainModuleTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_base_dir = main_module.BASE_DIR
        main_module.BASE_DIR = self.temp_dir.name
        main_module.models.clear()
        self.image = np.full((8, 8, 3), 127, dtype=np.uint8)

    def tearDown(self):
        main_module.BASE_DIR = self.original_base_dir
        main_module.models.clear()
        self.temp_dir.cleanup()

    def run_async(self, coroutine):
        return asyncio.run(coroutine)

    def make_upload(self):
        success, buffer = cv2.imencode(".jpg", self.image)
        self.assertTrue(success)
        return DummyUploadFile(buffer.tobytes())

    def test_startup_event_loads_all_models(self):
        rotation = mock.Mock(name="rotation")
        malboro = mock.Mock(name="malboro")
        computantis = mock.Mock(name="computantis")
        refiner = mock.Mock(name="refiner")
        pdf_engine = mock.Mock(name="pdf_engine")

        with mock.patch.object(main_module, "RotationDetector", return_value=rotation) as rotation_cls, \
             mock.patch.object(main_module, "MalboroDetector", return_value=malboro) as malboro_cls, \
             mock.patch.object(main_module, "ComputantisDetector", return_value=computantis) as computantis_cls, \
             mock.patch.object(main_module, "CornerBaneRefiner", return_value=refiner) as refiner_cls, \
             mock.patch.object(main_module, "PDFEngine", return_value=pdf_engine) as pdf_engine_cls:
            self.run_async(main_module.startup_event())

        rotation_cls.assert_called_once_with(output_dir=self.temp_dir.name)
        malboro_cls.assert_called_once_with(
            model_path=main_module.MODELS_CONFIG["malboro_path"],
            output_dir=self.temp_dir.name,
        )
        computantis_cls.assert_called_once_with(
            model_path=main_module.MODELS_CONFIG["computantis_path"],
            output_dir=self.temp_dir.name,
        )
        refiner_cls.assert_called_once_with(
            model_path=main_module.MODELS_CONFIG["refiner_path"],
            output_dir=self.temp_dir.name,
        )
        pdf_engine_cls.assert_called_once_with(font_path=main_module.MODELS_CONFIG["font_path"])
        self.assertEqual(
            main_module.models,
            {
                "rotation": rotation,
                "malboro": malboro,
                "computantis": computantis,
                "refiner": refiner,
                "pdf_engine": pdf_engine,
            },
        )

    def test_startup_event_exits_when_loading_fails(self):
        with mock.patch.object(main_module, "RotationDetector", side_effect=RuntimeError("boom")), \
             mock.patch("builtins.exit", side_effect=SystemExit(1)):
            with self.assertRaises(SystemExit) as error:
                self.run_async(main_module.startup_event())

        self.assertEqual(error.exception.code, 1)

    def test_load_image_from_upload_reads_valid_image(self):
        upload = self.make_upload()

        image = main_module.load_image_from_upload(upload)

        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape, self.image.shape)

    def test_load_image_from_upload_rejects_invalid_image(self):
        upload = DummyUploadFile(b"not-an-image")

        with self.assertRaises(main_module.HTTPException) as error:
            main_module.load_image_from_upload(upload)

        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(error.exception.detail, "Не удалось прочитать изображение")

    def test_get_request_dir_creates_directory(self):
        with mock.patch.object(main_module.uuid, "uuid4", return_value="request-123"):
            request_dir = main_module.get_request_dir()

        self.assertEqual(request_dir, str(Path(self.temp_dir.name) / "request-123"))
        self.assertTrue(Path(request_dir).is_dir())

    def test_enhance_document_returns_jpeg_response(self):
        enhancer = mock.Mock()
        enhancer.enhance.return_value = self.image

        with mock.patch.object(main_module, "load_image_from_upload", return_value=self.image), \
             mock.patch.object(main_module, "DocumentEnhancer", return_value=enhancer) as enhancer_cls:
            response = self.run_async(
                main_module.api_enhance_document(
                    file=object(),
                    brightness=1.1,
                    contrast=1.3,
                    whitening=0.8,
                    shadow_removal=False,
                    sharpen=True,
                )
            )

        enhancer_cls.assert_called_once_with(
            brightness=1.1,
            contrast=1.3,
            whitening=0.8,
            shadow_removal=False,
            sharpen=True,
        )
        enhancer.enhance.assert_called_once_with(self.image)
        self.assertEqual(response.media_type, "image/jpeg")
        self.assertGreater(len(response.body), 0)

    def test_enhance_document_wraps_encode_failure(self):
        enhancer = mock.Mock()
        enhancer.enhance.return_value = self.image

        with mock.patch.object(main_module, "load_image_from_upload", return_value=self.image), \
             mock.patch.object(main_module, "DocumentEnhancer", return_value=enhancer), \
             mock.patch.object(main_module.cv2, "imencode", return_value=(False, None)):
            with self.assertRaises(main_module.HTTPException) as error:
                self.run_async(main_module.api_enhance_document(file=object()))

        self.assertEqual(error.exception.status_code, 500)
        self.assertIn("Enhancement failed", error.exception.detail)
        self.assertIn("Failed to encode output image", error.exception.detail)

    def test_define_rotation_angle_returns_numbers(self):
        rotation = mock.Mock()
        rotation.detect_angle.return_value = (12.9, 0.87)
        main_module.models["rotation"] = rotation

        with mock.patch.object(main_module, "load_image_from_upload", return_value=self.image):
            payload = self.run_async(main_module.api_define_rotation_angle(file=object()))

        self.assertEqual(payload, {"angle": 12, "score": 0.87})
        rotation.detect_angle.assert_called_once_with(self.image)

    def test_find_corners_and_bbox_uses_malboro(self):
        malboro = mock.Mock()
        malboro.detect.return_value = (
            {"tl": (10.7, 20.3), "br": (99.9, 101.1)},
            [1.2, 2.8, 50.6, 60.4],
        )
        computantis = mock.Mock()
        main_module.models["malboro"] = malboro
        main_module.models["computantis"] = computantis

        with mock.patch.object(main_module, "load_image_from_upload", return_value=self.image):
            payload = self.run_async(main_module.api_find_corners_and_bbox(file=object()))

        self.assertEqual(payload["detector"], "malboro")
        self.assertEqual(payload["corners"], {"tl": (10, 20), "br": (99, 101)})
        self.assertEqual(payload["bbox"], [1, 2, 50, 60])
        computantis.detect.assert_not_called()

    def test_find_corners_and_bbox_falls_back_to_computantis(self):
        malboro = mock.Mock()
        malboro.detect.side_effect = RuntimeError("first detector failed")
        computantis = mock.Mock()
        computantis.detect.return_value = (
            {"tl": (1, 2), "tr": (3, 4)},
            [5, 6, 7, 8],
        )
        main_module.models["malboro"] = malboro
        main_module.models["computantis"] = computantis

        with mock.patch.object(main_module, "load_image_from_upload", return_value=self.image):
            payload = self.run_async(main_module.api_find_corners_and_bbox(file=object()))

        self.assertEqual(payload["detector"], "computantis")
        self.assertEqual(payload["bbox"], [5, 6, 7, 8])
        computantis.detect.assert_called_once_with(self.image)

    def test_find_corners_and_bbox_raises_when_all_detectors_fail(self):
        malboro = mock.Mock()
        malboro.detect.side_effect = RuntimeError("malboro failed")
        computantis = mock.Mock()
        computantis.detect.side_effect = RuntimeError("computantis failed")
        main_module.models["malboro"] = malboro
        main_module.models["computantis"] = computantis

        with mock.patch.object(main_module, "load_image_from_upload", return_value=self.image):
            with self.assertRaises(main_module.HTTPException) as error:
                self.run_async(main_module.api_find_corners_and_bbox(file=object()))

        self.assertEqual(error.exception.status_code, 500)
        self.assertIn("Detection failed on both models", error.exception.detail)

    def test_refine_corners_rejects_invalid_json(self):
        with mock.patch.object(main_module, "load_image_from_upload", return_value=self.image):
            with self.assertRaises(main_module.HTTPException) as error:
                self.run_async(main_module.api_refine_corners(file=object(), corners="oops"))

        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(error.exception.detail, "Invalid corners JSON format")

    def test_refine_corners_uses_provided_bbox(self):
        refiner = mock.Mock()
        refiner.refine.return_value = {"tl": (11.8, 22.2)}
        main_module.models["refiner"] = refiner
        corners = json.dumps({"tl": [10, 20], "br": [30, 40]})
        bbox = json.dumps([1, 2, 3, 4])

        with mock.patch.object(main_module, "load_image_from_upload", return_value=self.image):
            payload = self.run_async(
                main_module.api_refine_corners(file=object(), corners=corners, bbox=bbox)
            )

        refiner.refine.assert_called_once_with(
            self.image,
            coarse_corners={"tl": (10, 20), "br": (30, 40)},
            bbox=(1, 2, 3, 4),
        )
        self.assertEqual(payload, {"refined_corners": {"tl": (11, 22)}})

    def test_refine_corners_derives_bbox_when_bbox_is_invalid(self):
        refiner = mock.Mock()
        refiner.refine.return_value = {"tl": (1, 2), "br": (3, 4)}
        main_module.models["refiner"] = refiner
        corners = json.dumps(
            {
                "tl": [9, 20],
                "tr": [80, 22],
                "bl": [10, 100],
                "br": [81, 101],
            }
        )

        with mock.patch.object(main_module, "load_image_from_upload", return_value=self.image):
            payload = self.run_async(
                main_module.api_refine_corners(file=object(), corners=corners, bbox="not-json")
            )

        refiner.refine.assert_called_once_with(
            self.image,
            coarse_corners={
                "tl": (9, 20),
                "tr": (80, 22),
                "bl": (10, 100),
                "br": (81, 101),
            },
            bbox=(9, 20, 81, 101),
        )
        self.assertEqual(payload["refined_corners"]["br"], (3, 4))

    def test_refine_corners_wraps_refiner_errors(self):
        refiner = mock.Mock()
        refiner.refine.side_effect = RuntimeError("refine failed")
        main_module.models["refiner"] = refiner
        corners = json.dumps({"tl": [1, 2], "br": [3, 4]})

        with mock.patch.object(main_module, "load_image_from_upload", return_value=self.image):
            with self.assertRaises(main_module.HTTPException) as error:
                self.run_async(main_module.api_refine_corners(file=object(), corners=corners))

        self.assertEqual(error.exception.status_code, 500)
        self.assertIn("Refinement failed", error.exception.detail)

    def test_warp_perspective_rejects_invalid_json(self):
        with mock.patch.object(main_module, "load_image_from_upload", return_value=self.image):
            with self.assertRaises(main_module.HTTPException) as error:
                self.run_async(main_module.api_warp_perspective(file=object(), corners="bad"))

        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(error.exception.detail, "Invalid corners JSON format")

    def test_warp_perspective_returns_jpeg_response(self):
        corrector = mock.Mock()
        corrector.correct.return_value = self.image
        corners = json.dumps({"tl": [1, 2], "tr": [3, 4]})
        output_dir = str(Path(self.temp_dir.name) / "warp-dir")

        with mock.patch.object(main_module, "load_image_from_upload", return_value=self.image), \
             mock.patch.object(main_module, "get_request_dir", return_value=output_dir), \
             mock.patch.object(main_module, "HomographyCorrector", return_value=corrector) as corrector_cls:
            response = self.run_async(
                main_module.api_warp_perspective(file=object(), corners=corners)
            )

        corrector_cls.assert_called_once_with(output_dir=output_dir)
        corrector.correct.assert_called_once_with(
            self.image,
            {"tl": (1, 2), "tr": (3, 4)},
        )
        self.assertEqual(response.media_type, "image/jpeg")
        self.assertGreater(len(response.body), 0)

    def test_warp_perspective_wraps_encode_failure(self):
        corrector = mock.Mock()
        corrector.correct.return_value = self.image
        corners = json.dumps({"tl": [1, 2], "tr": [3, 4]})

        with mock.patch.object(main_module, "load_image_from_upload", return_value=self.image), \
             mock.patch.object(main_module, "HomographyCorrector", return_value=corrector), \
             mock.patch.object(main_module, "get_request_dir", return_value=self.temp_dir.name), \
             mock.patch.object(main_module.cv2, "imencode", return_value=(False, None)):
            with self.assertRaises(main_module.HTTPException) as error:
                self.run_async(main_module.api_warp_perspective(file=object(), corners=corners))

        self.assertEqual(error.exception.status_code, 500)
        self.assertIn("Warping failed", error.exception.detail)
        self.assertIn("Failed to encode output image", error.exception.detail)

    def test_do_ocr_reconstructs_pdf_with_image_dir(self):
        ocr = mock.Mock()
        pdf_engine = mock.Mock()
        main_module.models["pdf_engine"] = pdf_engine

        def recognize(_):
            json_path = Path(self.temp_dir.name) / "doc-1" / "ocr_output.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text("{}", encoding="utf-8")
            image_dir = json_path.parent / "ocr_output" / "imgs"
            image_dir.mkdir(parents=True, exist_ok=True)
            return str(json_path)

        def reconstruct(json_path, pdf_path, image_dir=None):
            Path(pdf_path).write_bytes(b"%PDF-1.4")

        ocr.recognize.side_effect = recognize
        pdf_engine.reconstruct.side_effect = reconstruct

        with mock.patch.object(main_module, "load_image_from_upload", return_value=self.image), \
             mock.patch.object(main_module, "OCRProcessor", return_value=ocr) as ocr_cls, \
             mock.patch.object(main_module.uuid, "uuid4", return_value="doc-1"):
            response = self.run_async(main_module.api_do_ocr(file=object()))

        ocr_cls.assert_called_once_with(output_dir=str(Path(self.temp_dir.name) / "doc-1"))
        self.assertEqual(response.path, str(Path(self.temp_dir.name) / "doc-1" / "output.pdf"))
        self.assertEqual(response.media_type, "application/pdf")
        pdf_engine.reconstruct.assert_called_once_with(
            str(Path(self.temp_dir.name) / "doc-1" / "ocr_output.json"),
            str(Path(self.temp_dir.name) / "doc-1" / "output.pdf"),
            image_dir=str(Path(self.temp_dir.name) / "doc-1" / "ocr_output" / "imgs"),
        )

    def test_do_ocr_reconstructs_pdf_without_image_dir(self):
        ocr = mock.Mock()
        pdf_engine = mock.Mock()
        main_module.models["pdf_engine"] = pdf_engine

        def recognize(_):
            json_path = Path(self.temp_dir.name) / "doc-2" / "ocr_output.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text("{}", encoding="utf-8")
            return str(json_path)

        def reconstruct(json_path, pdf_path, image_dir=None):
            self.assertIsNone(image_dir)
            Path(pdf_path).write_bytes(b"%PDF-1.4")

        ocr.recognize.side_effect = recognize
        pdf_engine.reconstruct.side_effect = reconstruct

        with mock.patch.object(main_module, "load_image_from_upload", return_value=self.image), \
             mock.patch.object(main_module, "OCRProcessor", return_value=ocr), \
             mock.patch.object(main_module.uuid, "uuid4", return_value="doc-2"):
            response = self.run_async(main_module.api_do_ocr(file=object()))

        self.assertEqual(response.path, str(Path(self.temp_dir.name) / "doc-2" / "output.pdf"))
        pdf_engine.reconstruct.assert_called_once_with(
            str(Path(self.temp_dir.name) / "doc-2" / "ocr_output.json"),
            str(Path(self.temp_dir.name) / "doc-2" / "output.pdf"),
        )

    def test_do_ocr_fails_when_json_is_missing(self):
        ocr = mock.Mock()
        ocr.recognize.return_value = str(Path(self.temp_dir.name) / "missing.json")
        main_module.models["pdf_engine"] = mock.Mock()

        with mock.patch.object(main_module, "load_image_from_upload", return_value=self.image), \
             mock.patch.object(main_module, "OCRProcessor", return_value=ocr), \
             mock.patch.object(main_module.uuid, "uuid4", return_value="doc-3"):
            with self.assertRaises(main_module.HTTPException) as error:
                self.run_async(main_module.api_do_ocr(file=object()))

        self.assertEqual(error.exception.status_code, 500)
        self.assertIn("OCR Pipeline failed", error.exception.detail)
        self.assertIn("OCR result JSON not found", error.exception.detail)

    def test_do_ocr_fails_when_pdf_is_not_generated(self):
        ocr = mock.Mock()
        pdf_engine = mock.Mock()
        main_module.models["pdf_engine"] = pdf_engine

        def recognize(_):
            json_path = Path(self.temp_dir.name) / "doc-4" / "ocr_output.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text("{}", encoding="utf-8")
            return str(json_path)

        ocr.recognize.side_effect = recognize

        with mock.patch.object(main_module, "load_image_from_upload", return_value=self.image), \
             mock.patch.object(main_module, "OCRProcessor", return_value=ocr), \
             mock.patch.object(main_module.uuid, "uuid4", return_value="doc-4"):
            with self.assertRaises(main_module.HTTPException) as error:
                self.run_async(main_module.api_do_ocr(file=object()))

        self.assertEqual(error.exception.status_code, 500)
        self.assertIn("OCR Pipeline failed", error.exception.detail)
        self.assertIn("PDF was not generated", error.exception.detail)

    def test_stretch_to_aspect_returns_jpeg_response(self):
        with mock.patch.object(main_module, "load_image_from_upload", return_value=self.image):
            response = self.run_async(
                main_module.api_stretch_to_aspect(
                    file=object(),
                    target_width=16,
                    target_height=9,
                )
            )

        self.assertEqual(response.media_type, "image/jpeg")
        self.assertGreater(len(response.body), 0)

    def test_stretch_to_aspect_wraps_encoding_failure(self):
        with mock.patch.object(main_module, "load_image_from_upload", return_value=self.image), \
             mock.patch.object(main_module.cv2, "imencode", return_value=(False, None)):
            with self.assertRaises(main_module.HTTPException) as error:
                self.run_async(
                    main_module.api_stretch_to_aspect(
                        file=object(),
                        target_width=16,
                        target_height=9,
                    )
                )

        self.assertEqual(error.exception.status_code, 500)
        self.assertIn("Encoding failed", error.exception.detail)


if __name__ == "__main__":
    unittest.main(verbosity=2)
