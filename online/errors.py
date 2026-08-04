"""Typed application errors mapped to stable API error codes."""


class OnlineError(Exception):
    code = "online_error"


class ConfigurationError(OnlineError):
    code = "configuration_error"


class MetadataNotFoundError(OnlineError):
    code = "metadata_not_found"


class DependencyUnavailableError(OnlineError):
    code = "dependency_unavailable"


class InvalidQueryError(OnlineError):
    code = "invalid_query"


class TaskConflictError(OnlineError):
    """Body khai báo task khác với task của endpoint.

    Trước PR-01 route ghi đè `request.task` bằng task của path một cách im
    lặng, nên client gửi nhầm không bao giờ biết. Giờ là lỗi tường minh.
    """

    code = "task_conflict"

    def __init__(self, body_task: str, path_task: str) -> None:
        super().__init__(
            f"body task {body_task!r} conflicts with endpoint task {path_task!r}; "
            "omit `task` in the body or call POST /v1/search instead"
        )
        self.body_task = body_task
        self.path_task = path_task

