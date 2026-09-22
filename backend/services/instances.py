import threading
import logging

logger = logging.getLogger(__name__)

_paddle_ocr = None
_llm = None
_crnn = None
_lock = threading.Lock()


def get_paddle_ocr():
    global _paddle_ocr
    if _paddle_ocr is None:
        with _lock:
            if _paddle_ocr is None:
                from services.paddle_service import PaddleOCRService
                logger.info("Initializing PaddleOCRService on demand...")
                _paddle_ocr = PaddleOCRService()
    return _paddle_ocr


def get_llm():
    global _llm
    if _llm is None:
        with _lock:
            if _llm is None:
                from services.llm_service import LLMService
                logger.info("Initializing LLMService on demand...")
                _llm = LLMService()
    return _llm


def get_crnn():
    global _crnn
    if _crnn is None:
        with _lock:
            if _crnn is None:
                from services.crnn_service import CRNNService
                logger.info("Initializing CRNNService on demand...")
                _crnn = CRNNService()
    return _crnn


class _LazyProxy:
    """Proxy object that delegates all attribute lookups to the singleton instance upon first use."""
    def __init__(self, getter):
        object.__setattr__(self, "_getter", getter)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_getter")(), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_getter")(), name, value)


# Exported proxies — server starts and binds port immediately, models load on demand
paddle_ocr = _LazyProxy(get_paddle_ocr)
llm = _LazyProxy(get_llm)
crnn = _LazyProxy(get_crnn)
