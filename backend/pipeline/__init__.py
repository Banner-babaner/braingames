from .preprocess import preprocess_image
from .ocr import extract_text
from .chunk import chunk_text

def run_pipeline(image_bytes: bytes, use_canny: bool = False) -> tuple[str, list[str]]:
    """Байты → нормализация → OCR → чанки."""
    image = preprocess_image(image_bytes, use_canny=use_canny)
    text = extract_text(image)
    chunks = chunk_text(text)
    return text, chunks