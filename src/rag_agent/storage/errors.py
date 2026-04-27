class StorageError(Exception):
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


class IndexLoadError(StorageError):
    def __init__(self, detail: str = "Vector index could not be loaded.") -> None:
        super().__init__(
            detail,
            status_code=500,
            error_code="index_load_error",
        )


class IndexWriteError(StorageError):
    def __init__(self, detail: str = "Vector index could not be written.") -> None:
        super().__init__(
            detail,
            status_code=500,
            error_code="index_write_error",
        )
