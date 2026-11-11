from .preprocess import preprocess_image
from .ocr import extract_text
from .chunk import chunk_text

def run_pipeline(image_bytes: bytes) -> tuple[str, list[str]]:
    """Полный пайплайн: байты → нормализация → OCR → чанки."""
    image = preprocess_image(image_bytes)
    text = extract_text(image)
    chunks = chunk_text(text)
    return text, chunks