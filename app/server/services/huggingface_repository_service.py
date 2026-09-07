from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


###############################################################################
class HuggingFaceReferenceError(ValueError):
    pass


###############################################################################
class HuggingFaceUnavailableError(RuntimeError):
    pass


###############################################################################
@dataclass(frozen=True, slots=True)
class HuggingFaceRepositoryMetadata:
    commit: str
    library_name: str | None
    pipeline_tag: str | None


###############################################################################
class HuggingFaceRepositoryService:

    # -------------------------------------------------------------------------
    def __init__(self, timeout_seconds: float = 6.0) -> None:
        self._timeout_seconds = timeout_seconds

    # -------------------------------------------------------------------------
    def resolve(self, repository_id: str, revision: str) -> HuggingFaceRepositoryMetadata:
        encoded_repo = quote(repository_id, safe="/")
        encoded_revision = quote(revision, safe="")
        url = f"https://huggingface.co/api/models/{encoded_repo}/revision/{encoded_revision}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "NeuroWeave/1.0 custom-node-reference-resolver",
            },
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                decoded = response.read().decode("utf-8")
                raw_payload = cast(object, json.loads(decoded))
        except HTTPError as exc:
            if exc.code in {401, 403, 404}:
                raise HuggingFaceReferenceError(
                    "Hugging Face repository or revision was not found or is not publicly accessible"
                ) from exc
            raise HuggingFaceUnavailableError(
                f"Hugging Face returned HTTP {exc.code} while resolving the reference"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise HuggingFaceUnavailableError(
                "Hugging Face could not be reached to verify the public repository reference"
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise HuggingFaceUnavailableError(
                "Hugging Face returned an unreadable repository response"
            ) from exc

        if not isinstance(raw_payload, dict):
            raise HuggingFaceUnavailableError(
                "Hugging Face returned an unexpected repository response"
            )
        payload = cast(dict[str, object], raw_payload)
        commit = payload.get("sha")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
            raise HuggingFaceUnavailableError(
                "Hugging Face did not return a full resolved commit for this revision"
            )
        library_name = payload.get("library_name")
        pipeline_tag = payload.get("pipeline_tag")
        return HuggingFaceRepositoryMetadata(
            commit=commit.casefold(),
            library_name=library_name if isinstance(library_name, str) else None,
            pipeline_tag=pipeline_tag if isinstance(pipeline_tag, str) else None,
        )
