"""PyMuPDF is not thread-safe, and uploads and literature checks run side by side: paths that read PDFs in worker
threads hold this lock."""
import threading

PYMUPDF = threading.Lock()
