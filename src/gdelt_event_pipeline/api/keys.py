"""Re-export from routers.keys for backward compatibility."""

from gdelt_event_pipeline.api.routers.keys import (  # noqa: F401
    KeyCreated,
    KeyMeta,
    _generate_key,
    auth_config,
    create_key,
    get_key,
    revoke_key,
    router,
)
from gdelt_event_pipeline.storage.database import get_pool  # noqa: F401
