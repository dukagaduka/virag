from __future__ import annotations

import asyncio

import httpx

from virag.errors import ApiError
from virag.settings import EmbeddingsSettings


class EmbeddingsClient:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str,
        timeout: float | httpx.Timeout,
        chunk_size: int = 16,
    ):

        self.base_url   = base_url
        self.model_name = model_name
        self.api_key    = api_key
        self.timeout    = timeout
        self.chunk_size = chunk_size


    @property
    def route(self) -> str:
        return f"{self.base_url}/embeddings"


    @property
    def models_route(self) -> str:
        return f"{self.base_url}/models"


    @property
    def auth_header(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}


    @classmethod
    def from_settings(cls, es: EmbeddingsSettings) -> EmbeddingsClient:
        return cls(
            base_url=es.base_url,
            model_name=es.model_name,
            api_key=es.api_key,
            timeout=es.timeout,
        )


    async def check_connection(self) -> None:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    self.models_route,
                    headers=self.auth_header,
                )
                response.raise_for_status()

            except httpx.HTTPStatusError as e:
                raise ApiError(
                    status_code=503,
                    message=f"Cannot establish connection to the {self.models_route} during the check",
                    details={"http_status_error": {"code": e.response.status_code}, "text": e.response.text},
                ) from e

            except httpx.RequestError as e:
                raise ApiError(
                    status_code=503,
                    message=f"Cannot reach {self.models_route} during the check",
                    details={"error": repr(e)},
                ) from e

            try:
                data = response.json()
                models = [model["id"] for model in data["data"]]

            except (KeyError, TypeError, ValueError) as e:
                raise ApiError(
                    status_code=422,
                    message=f"Cannot decode {self.models_route} response or unexpected response structure during the check",
                    details={"base_message": str(e)},
                ) from e

            if self.model_name not in models:
                raise ApiError(
                    status_code=404,
                    message=f"Model {self.model_name} is not established at {self.base_url}",
                    details={"models": models},
                )


    async def _post(
        self,
        client: httpx.AsyncClient,
        body: dict,
    ) -> list[list[float]]:
        """POST one embeddings request and return its vectors ordered by `index`.

        Transport is shared by both flows; only the request body differs — text uses
        `input`, images use chat-style `messages` (see `embed_texts`/`embed_images`).
        """
        try:
            response = await client.post(
                self.route,
                json={"model": self.model_name, **body},
                headers=self.auth_header,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            raise ApiError(
                status_code=503,
                message=f"Cannot establish connection to the {self.route} during embedding",
                details={"httpx_status_error": {"code": e.response.status_code}, "text": e.response.text},
            ) from e

        except httpx.RequestError as e:
            raise ApiError(
                status_code=503,
                message=f"Cannot reach {self.route} during embedding",
                details={"error": repr(e)},
            ) from e

        try:
            data = response.json()
            embeddings = sorted(data["data"], key=lambda x: x["index"])

            return [emb["embedding"] for emb in embeddings]

        except (KeyError, TypeError, ValueError) as e:
            raise ApiError(
                status_code=422,
                message=f"Cannot decode {self.route} response or unexpected response structure during embedding",
                details={"base_message": str(e)},
            ) from e


    async def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Embed query/text inputs. Sent through the `input` field, batched by chunk_size."""
        all_embeddings: list[list[float]] = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for i in range(0, len(texts), self.chunk_size):
                chunk = texts[i:i + self.chunk_size]
                all_embeddings.extend(await self._post(client, {"input": chunk}))

        return all_embeddings


    async def embed_images(
        self,
        base64_images: list[str],
        *,
        image_mime: str = "image/jpeg",
    ) -> list[list[float]]:
        """Embed base64 tile images.

        vLLM's chat-embeddings takes an image only via a `messages`/`image_url`
        payload, and one message list yields exactly one vector — so this is one
        request per image, fired `chunk_size` at a time. Output order matches input.
        """
        def image_body(b64: str) -> dict:
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{image_mime};base64,{b64}"},
                            }
                        ],
                    }
                ]
            }

        all_embeddings: list[list[float]] = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for i in range(0, len(base64_images), self.chunk_size):
                group = base64_images[i:i + self.chunk_size]
                results = await asyncio.gather(
                    *(self._post(client, image_body(b64)) for b64 in group)
                )

                for vectors in results:
                    all_embeddings.extend(vectors)

        return all_embeddings
