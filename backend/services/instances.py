from services.paddle_service import PaddleOCRService
from services.llm_service import LLMService
from services.crnn_service import CRNNService

print("Initializing shared OCR and LLM services...")

paddle_ocr = PaddleOCRService()
llm = LLMService()
crnn = CRNNService()

print("Shared services initialized!")
