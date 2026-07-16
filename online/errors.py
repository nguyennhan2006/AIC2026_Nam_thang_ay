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

