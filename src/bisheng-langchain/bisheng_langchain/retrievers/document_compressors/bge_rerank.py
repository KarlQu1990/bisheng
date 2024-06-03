from __future__ import annotations

import requests
from typing import Dict, Optional, Sequence, Any, Union, Tuple
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain.utils import get_from_dict_or_env
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor


class BGERerank(BaseDocumentCompressor):
    """Document compressor using bge-rerank interface."""

    client: Optional[Any]
    top_n: int = 3
    """Number of documents to return."""
    model: Optional[str] = None
    """Model to use for reranking."""
    host_base_url: str = None
    url_ep: Optional[str] = None
    request_timeout: Optional[Union[float, Tuple[float, float]]] = 200
    """Timeout in seconds for the OpenAPI request."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["host_base_url"] = get_from_dict_or_env(values, "host_base_url", "HostBaseUrl")
        model = values["model"]
        try:
            url = values["host_base_url"]
            values["url_ep"] = f"{url}/{model}/infer"
        except Exception:
            raise Exception(f"Failed to set url ep failed for model {model}")

        try:
            values["client"] = requests.post
        except AttributeError:
            raise ValueError("Try upgrading it with `pip install --upgrade requests`.")
        return values

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:

        inp_local = {
            "query": query,
            "texts": [doc.page_content for doc in documents],
            "model": self.model,
        }

        resp = self.client(url=self.url_ep, json=inp_local, timeout=self.request_timeout).json()

        scores = resp["scores"]
        sorted_results = sorted(zip(range(len(documents)), documents, scores), key=lambda x: x[-1], reverse=True)

        final_results = []
        for idx, doc, score in sorted_results[: self.top_n]:
            doc = Document(
                page_content=doc.page_content,
                metadata={"id": idx, "relevance_score": score},
            )
            final_results.append(doc)

        return final_results
