import os
from dataclasses import dataclass
from itertools import combinations, permutations

import pandas as pd
import pytest

import tests.conftest
from eda_ja.eda import EdaJa, Words


@pytest.fixture
def eda(mocker):
    df = pd.DataFrame(
        [
            (1, "明日"),
            (1, "次の日"),
            (2, "昨日"),
        ],
        columns=["synset", "lemma"],
    )
    mocker.patch.object(EdaJa, "_create_wordnet", return_value=df)
    return EdaJa(stop_words_path=None, wordnet_path=None)


def test_get_synonyms(eda):
    result = eda._get_synonyms("明日")
    assert sorted(result)[0] == "次の日"


def test_synonym_replacement(eda):
    data = ["明日", "は", "晴れ"]
    result = eda.synonym_replacement(Words(data, data), 1)
    assert result == ["次の日", "は", "晴れ"]


def test_random_deletion_one_word(eda):
    result = eda.random_deletion(["明日"], 1.0)
    assert result == ["明日"]


def test_random_deleteion_all(eda):
    result = eda.random_deletion(["明日", "は", "晴れ"], 1.0)
    assert len(result) == 1
    assert result[0] in ["明日", "は", "晴れ"]


def test_random_deletion(eda):
    data = ["明日", "は", "晴れ"]
    result = eda.random_deletion(data, 0.5)
    expected = (
        list(combinations(data, 1))
        + list(combinations(data, 2))
        + list(combinations(data, 3))
    )
    assert tuple(result) in expected


def test_random_swap(eda):
    data = ["明日", "は", "晴れ"]
    result = eda.random_swap(data, 1)
    assert result in [
        ["は", "明日", "晴れ"],
        ["明日", "晴れ", "は"],
        ["晴れ", "は", "明日"],
    ]


def test_random_insertion_no_synonym(eda):
    data = ["昨日", "は", "晴れ"]
    result = eda.random_insertion(data, 1)
    assert result == data


def test_random_insertion(eda):
    data = ["明日", "は", "晴れ"]
    result = eda.random_insertion(data, 1)
    expected = [
        ["次の日", "明日", "は", "晴れ"],
        ["明日", "次の日", "は", "晴れ"],
        ["明日", "は", "次の日", "晴れ"],
    ]
    assert result in expected
