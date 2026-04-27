class OllamaError(Exception):
    def __init__(
        self,
        detail: str,
        *,
        status_code: int,
        error_code: str,
        upstream_status_code: int | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code
        self.error_code = error_code
        self.upstream_status_code = upstream_status_code


class OllamaUnavailableError(OllamaError):
    def __init__(self, detail: str = "Ollama is unavailable.") -> None:
        super().__init__(
            detail,
            status_code=503,
            error_code="ollama_unavailable",
        )


class OllamaTimeoutError(OllamaError):
    def __init__(self, detail: str = "Ollama request timed out.") -> None:
        super().__init__(
            detail,
            status_code=504,
            error_code="ollama_timeout",
        )


class OllamaResponseError(OllamaError):
    def __init__(
        self,
        detail: str = "Ollama returned an invalid response.",
        *,
        upstream_status_code: int | None = None,
    ) -> None:
        super().__init__(
            detail,
            status_code=502,
            error_code="ollama_response_error",
            upstream_status_code=upstream_status_code,
        )
