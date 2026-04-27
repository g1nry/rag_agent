class DocumentIngestionError(Exception):
    def __init__(
        self,
        detail: str,
        *,
        status_code: int,
        error_code: str,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code
        self.error_code = error_code


class EmptyDocumentError(DocumentIngestionError):
    def __init__(self, detail: str = "Uploaded document is empty.") -> None:
        super().__init__(
            detail,
            status_code=400,
            error_code="empty_document",
        )


class DocumentTooLargeError(DocumentIngestionError):
    def __init__(self, limit_bytes: int) -> None:
        super().__init__(
            f"Uploaded document exceeds the maximum allowed size of {limit_bytes} bytes.",
            status_code=413,
            error_code="document_too_large",
        )


class UnsupportedDocumentTypeError(DocumentIngestionError):
    def __init__(self, filename: str) -> None:
        super().__init__(
            f"Unsupported document type for file '{filename}'.",
            status_code=415,
            error_code="unsupported_document_type",
        )


class DocumentEncodingError(DocumentIngestionError):
    def __init__(self, filename: str) -> None:
        super().__init__(
            f"File '{filename}' must be UTF-8 encoded text.",
            status_code=400,
            error_code="document_encoding_error",
        )
