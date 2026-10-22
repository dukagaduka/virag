from __future__ import annotations

import asyncio

from virag.clients.http_service import OpenAIService
from virag.settings import HttpSettings, RetrieverSettings


class RetrieverClient(OpenAIService):
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str,
        *,
        chunk_size: int = 16,
        **http_options,
    ):

        super().__init__(base_url, model_name, api_key, **http_options)

        self.chunk_size = chunk_size


    @property
    def route(self) -> str:
        return f"{self.base_url}/embeddings"


    @classmethod
    def from_settings(cls, rs: RetrieverSettings, hs: HttpSettings) -> RetrieverClient:
        return cls(
            base_url=rs.base_url,
            model_name=rs.model_name,
            api_key=rs.api_key,
            chunk_size=rs.chunk_size,
            timeout=rs.timeout,
            connect_timeout=rs.connect_timeout,
            max_connections=rs.max_connections,
            max_retries=rs.max_retries,
            backoff_base=hs.backoff_base,
            backoff_cap=hs.backoff_cap,
        )


    async def _post(self, body: dict) -> list[list[float]]:
        """POST one embeddings request and return its vectors ordered by `index`.

        Transport is shared by both flows; only the request body differs — text uses
        `input`, images use chat-style `messages` (see `embed_texts`/`embed_images`).
        """
        response = await self._request(
            "POST",
            self.route,
            action="embedding",
            json={"model": self.model_name, **body},
        )

        return self._json(response, action="embedding", parse=self._vectors)


    async def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Embed query/text inputs. Sent through the `input` field, batched by chunk_size."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.chunk_size):
            chunk = texts[i:i + self.chunk_size]
            all_embeddings.extend(await self._post({"input": chunk}))

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

        for i in range(0, len(base64_images), self.chunk_size):
            group = base64_images[i:i + self.chunk_size]
            results = await asyncio.gather(
                *(self._post(image_body(b64)) for b64 in group)
            )

            for vectors in results:
                all_embeddings.extend(vectors)

        return all_embeddings


    @staticmethod
    def _vectors(data: dict) -> list[list[float]]:
        embeddings = sorted(data["data"], key=lambda x: x["index"])

        return [emb["embedding"] for emb in embeddings]
