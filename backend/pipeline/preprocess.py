import cv2
import numpy as np

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Нормализация изображения: удаление теней, выравнивание фона, Deskewing."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось декодировать изображение")

    # 1. Переводим в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Убираем тени с помощью морфологических операций (Background Subtraction)
    # Создаем "фон" с помощью дилатации, а затем вычитаем его из оригинала
    kernel = np.ones((7, 7), np.uint8)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=5)
    # Вычитаем фон, чтобы выровнять освещение
    diff = cv2.absdiff(gray, background)
    
    # 3. Инвертируем и применяем порог Оцу (Otsu), чтобы получить контрастный текст
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Инвертируем обратно (черный текст на белом фоне)
    thresh = cv2.bitwise_not(thresh)

    # 4. Выравнивание наклона (Deskewing)
    inv_thresh = cv2.bitwise_not(thresh)
    coords = np.column_stack(np.where(inv_thresh > 0))
    
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 45:
            (h, w) = thresh.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated

    return thresh