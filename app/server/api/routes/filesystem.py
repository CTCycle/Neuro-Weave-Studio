from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import get_filesystem_service
from domain.filesystem import FilesystemBrowseResponse
from services.filesystem_service import FilesystemService

router = APIRouter(prefix="/api/filesystem", tags=["filesystem"])

###############################################################################
@router.get("/browse", response_model=FilesystemBrowseResponse)
def browse_filesystem(
    path: str = Query(default=""),
    include_files: bool = Query(default=True),
    extensions: str = Query(default=""),
    service: FilesystemService = Depends(get_filesystem_service),
) -> FilesystemBrowseResponse:
    normalized_extensions = tuple(
        item.strip() for item in extensions.split(",") if item.strip()
    )
    try:
        return service.browse(
            path,
            include_files=include_files,
            extensions=normalized_extensions,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except NotADirectoryError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to browse path: {exc}",
        ) from exc
