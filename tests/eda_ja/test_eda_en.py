import pytest
from dataclasses import dataclass
from nltk.corpus import wordnet 

import tests.conftest
from eda_ja.eda import EdaEn, Words


@pytest.fixture
def eda(mocker):
    return EdaEn(stop_words_path=None)

def test_synonym_replacement(mocker, eda):
    def mock_get_synonyms(word):
        if word == "sunny":
            return ["rainy"]
        return []

    mocker.patch.object(eda, "_get_synonyms", side_effect=mock_get_synonyms)

    data = ["It", "is", "sunny"]
    result = eda.synonym_replacement(Words(data, data), 1)
    assert result == ["It", "is", "rainy"]
