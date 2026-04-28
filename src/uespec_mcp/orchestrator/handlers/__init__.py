from .assertion_failed import propose_patch_for_assertion_failed
from .compile_error import propose_patch_for_compile_error
from .slate_event_not_handled import propose_patch_for_slate_event_not_handled
from .timeout import propose_patch_for_timeout
from .widget_missing import propose_patch_for_widget_missing

__all__ = [
    "propose_patch_for_assertion_failed",
    "propose_patch_for_compile_error",
    "propose_patch_for_slate_event_not_handled",
    "propose_patch_for_timeout",
    "propose_patch_for_widget_missing",
]
