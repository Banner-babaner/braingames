import pytesseract
import numpy as np

def extract_text(image: np.ndarray, lang: str = "rus+eng") -> str:
    """OCR через Tesseract. PSM 3 — автоматический анализ структуры страницы."""
    # --oem 3: использовать LSTM нейросеть
    # --psm 3: полностью автоматический режим (ищет блоки, абзацы, таблицы)
    config = "--oem 3 --psm 3" 
    text = pytesseract.image_to_string(image, lang=lang, config=config)
    return text.strip()