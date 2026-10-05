from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.dependencies import ServiceContainer
from app.main import create_app
from src.nlp.chunking import Chunk
from tests.helpers import FakeLLM, HashingEncoder, make_settings


@pytest.fixture()
def encoder() -> HashingEncoder:
    return HashingEncoder(dimension=512)


@pytest.fixture()
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture()
def sample_chunks() -> list[Chunk]:
    texts = {
        "plants.txt": "Photosynthesis lets plants convert sunlight, water and carbon dioxide into glucose using chlorophyll in the leaves.",
        "engines.txt": "A diesel engine compresses air until it is hot enough to ignite injected fuel without a spark plug.",
        "oceans.txt": "Ocean tides are caused mainly by the gravitational pull of the moon on seawater.",
        "bakery.txt": "Sourdough bread rises because wild yeast and bacteria ferment the flour and release carbon dioxide.",
    }
    return [
        Chunk(chunk_id=f"{doc}-0000", document=doc, text=text, source_path=f"/data/{doc}", chunk_index=0, word_count=len(text.split()))
        for doc, text in texts.items()
    ]


@pytest.fixture()
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture()
def container(settings, encoder, llm) -> ServiceContainer:
    return ServiceContainer(settings, encoder=encoder, llm=llm, intent_predictor=None)


@pytest.fixture()
def client(settings, container):
    app = create_app(settings, container)
    with TestClient(app) as test_client:
        yield test_client
