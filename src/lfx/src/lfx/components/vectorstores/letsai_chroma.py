from copy import deepcopy
import json

from chromadb.config import Settings
from langchain_chroma import Chroma
from typing_extensions import override

from langflow.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from langflow.base.vectorstores.utils import chroma_collection_to_data
from langflow.components.vectorstores import ChromaVectorStoreComponent
from langflow.io import BoolInput, DropdownInput, FloatInput, HandleInput, IntInput, MultilineInput, StrInput
from langflow.schema import Data, DataFrame


class LetsAIChromaVectorStoreComponent(ChromaVectorStoreComponent):
    """Custom Chroma Vector Store with enhanced search capabilities, including similarity+score and metadata filtering."""

    display_name: str = "LetsAI Chroma DB"
    description: str = "Chroma Vector Store with search capabilities and metadata filtering"
    name = "LetsaiChroma"
    icon = "Chroma"

    inputs = ChromaVectorStoreComponent.inputs + [
        FloatInput(
            name="sim_threshold",
            display_name="Similarity Threshold",
            info="Minimum similarity score (0 to 1) to include a document. Only applies to 'Similarity with Score' search.",
            value=0.0,
        ),
        MultilineInput(
            name="search_filter",
            display_name="Advanced Search Filter",
            info="Dictionary of metadata filters to refine search results.",
            tool_mode=True,
        ),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Update search_type options to include "Similarity with Score"
        for input_field in self.inputs:
            if input_field.name == "search_type":
                input_field.options = ["Similarity", "Similarity with Score", "MMR"]
                input_field.value = "Similarity"
                break

    @override
    def set_attributes(self, params: dict):
        super().set_attributes(params)
        raw_filter = params.get("search_filter", "")
        if raw_filter:
            try:
                self.advance_search_filter = json.loads(raw_filter)
            except json.JSONDecodeError:
                raise ValueError("The 'search_filter' must be a valid JSON dictionary.")
        else:
            self.advance_search_filter = None

        self.simil_threshold = params.get("sim_threshold", 0.0)

    @override
    def search_documents(self) -> list[Data]:
        """Search for documents in the vector store, with optional score and metadata filter."""
        if self._cached_vector_store is not None:
            vs = self._cached_vector_store
        else:
            vs = self.build_vector_store()
            self._cached_vector_store = vs

        query = self.search_query
        if not query:
            self.status = ""
            return []

        mode = self.search_type
        k = self.number_of_results
        filt = self.advance_search_filter
        threshold = float(self.simil_threshold or 0.0)

        if mode == "Similarity with Score" and hasattr(vs, "similarity_search_with_score"):
            docs_and_scores = vs.similarity_search_with_relevance_scores(
                query, k=k, filter=filt if filt else None
            )
            results: list[Data] = []
            for doc, score in docs_and_scores:
                if score >= threshold:
                    data = Data(
                        metadata={**getattr(doc, "metadata", {})},
                        score={"score": score},
                        text=doc.page_content,
                    )
                    results.append(data)
            self.status = results
            return results

        if filt:
            self.log(f"Filter: {filt}")
        self.log(f"Search input: {query}")
        self.log(f"Search type: {mode}")
        self.log(f"Number of results: {k}")
        self.log(f"Similarity threshold: {threshold}")

        if mode.lower() in ["similarity", "mmr"]:
            if hasattr(vs, "search"):
                search_args = {
                    "query": query,
                    "search_type": mode.lower(),
                    "k": k,
                }
                if filt:
                    search_args["filter"] = filt
                docs = vs.search(**search_args)
                data_list = [Data(metadata={**getattr(d, "metadata", {})}, text=d.page_content) for d in docs]
                self.status = data_list
                return data_list

        return super().search_documents()

    def as_dataframe(self) -> DataFrame:
        return DataFrame(self.search_documents())