"""Interactive Web UI route: GET / and GET /ui."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"], include_in_schema=False)

_UI_HTML_PATH = Path(__file__).resolve().parent.parent.parent / "ui" / "index.html"


@router.get("/", response_class=HTMLResponse)
@router.get("/ui", response_class=HTMLResponse)
def get_ui() -> HTMLResponse:
    """Serve the single-page interactive chat & document management UI."""
    if _UI_HTML_PATH.is_file():
        content = _UI_HTML_PATH.read_text(encoding="utf-8")
        return HTMLResponse(content=content)
    return HTMLResponse(
        content="<h1>UI template not found</h1>", status_code=404
    )
