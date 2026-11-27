from fastapi import HTTPException
import database as db

def check_subscription(user_id: str, file_size: int) -> None:
    """
    403 — нет права пользоваться сервисом (нет подписки, истекла).
    402 — подписка есть, но лимит исчерпан.
    """
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(403, "Нет активной подписки. Оформите тариф.")

    if user["expires_at"] < db._now():
        raise HTTPException(403, "Подписка истекла. Продлите тариф.")

    limits = db.PLAN_LIMITS.get(user["plan"])
    if not limits:
        raise HTTPException(403, f"Неизвестный тариф '{user['plan']}'")

    file_mb = file_size / (1024 * 1024)
    if file_mb > limits["max_file_mb"]:
        raise HTTPException(
            402,
            f"Файл {file_mb:.1f} МБ превышает лимит тарифа '{user['plan']}' "
            f"({limits['max_file_mb']} МБ)"
        )

    if limits["docs_per_month"] != -1:
        used = db.count_docs_this_month(user_id)
        if used >= limits["docs_per_month"]:
            raise HTTPException(
                402,
                f"Лимит документов исчерпан ({used}/{limits['docs_per_month']}). "
                f"Перейдите на тариф выше."
            )

def get_owned_document(user_id: str, doc_id: str) -> dict:
    """404, а не 403 — не раскрываем существование чужого документа."""
    doc = db.get_document(doc_id)
    if not doc or doc["owner_id"] != user_id:
        raise HTTPException(404, "Документ не найден")
    return doc