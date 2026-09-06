import hashlib
from pathlib import Path


def calculate_file_hash(path: Path) -> str:
    """
    Calculate SHA-256 hash of a file.

    The hash changes whenever the file content changes.
    """

    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def get_file_type(path: Path) -> str:
    """
    Return a normalized file type from the file extension.
    """

    extension = path.suffix.lower()

    file_types = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".xlsx": "xlsx",
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".webp": "image",
    }

    return file_types.get(extension, "unknown")


def is_supported_file(path: Path) -> bool:
    """
    Check whether the file can currently be processed by the RAG.
    """

    supported_extensions = {
        ".pdf",
        ".docx",
        ".xlsx",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }

    return path.suffix.lower() in supported_extensions