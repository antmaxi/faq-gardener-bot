from gardener.cluster import fingerprint, greedy_clusters
from gardener.graph import compile_graph
from gardener.store import connect, mark, seen
import numpy as np


class ScriptLLM:
    def complete(self, system: str, user: str) -> str:
        if "Task: grade" in user:
            if "лыж" in user:
                return '{"label": "not_a_faq"}'
            return '{"label": "missing"}'
        if "Task: draft" in user:
            assert "лыж" not in user
            return (
                '{"heading": "Семейная страховка", '
                '"outline": "- TODO: какие кассы принимают семьи\\n- TODO: какие документы нужны"}'
            )
        raise AssertionError(user)


class FakeIndex:
    def embed_queries(self, texts):
        rows = []
        for text in texts:
            if "страхов" in text:
                rows.append([1.0, 0.0, 0.0])
            elif "лыж" in text:
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([0.0, 0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)

    def retrieve(self, query: str, k: int = 4):
        from gardener.faq_index import FaqHit

        if "страхов" in query:
            return [
                FaqHit(
                    heading="Страховка · Семья",
                    text="Семейную страховку оформляют отдельно.",
                    url="https://ru-ch.github.io/faq/docs/Страховка.html",
                    score=1.0,
                )
            ]
        return []


QUESTIONS = [
    {"ts": 4, "text": "как оформить семейную страховку в Цюрихе"},
    {"ts": 3, "text": "как оформить семейную страховку для детей"},
    {"ts": 2, "text": "сколько стоила поездка на лыжах в феврале"},
    {"ts": 1, "text": "сколько стоила поездка на лыжах в прошлом году"},
]


def test_greedy_clusters_keeps_matching_pairs():
    vectors = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    groups = greedy_clusters(vectors, 0.9)
    assert sorted(len(group) for group in groups) == [2, 2]


def test_graph_drafts_only_the_wiki_gap(tmp_path):
    conn = connect(tmp_path / "g.db")
    graph = compile_graph(
        index=FakeIndex(),
        llm=ScriptLLM(),
        conn=conn,
        min_cluster_size=2,
        cluster_cosine=0.82,
        top_k=4,
    )
    result = graph.invoke({"questions": QUESTIONS})
    assert len(result["drafts"]) == 1
    draft = result["drafts"][0]
    assert draft["label"] == "missing"
    assert draft["heading"] == "Семейная страховка"
    assert draft["nearest_url"] == "https://ru-ch.github.io/faq/docs/Страховка.html"
    assert len(draft["questions"]) == 2
    assert "TODO" in draft["outline"]
    assert "лыж" not in draft["heading"]
    conn.close()


def test_already_sent_cluster_is_skipped(tmp_path):
    conn = connect(tmp_path / "g.db")
    insurance = [
        "как оформить семейную страховку в Цюрихе",
        "как оформить семейную страховку для детей",
    ]
    mark(conn, fingerprint(insurance), "Семейная страховка", insurance)
    graph = compile_graph(
        index=FakeIndex(),
        llm=ScriptLLM(),
        conn=conn,
        min_cluster_size=2,
        cluster_cosine=0.82,
        top_k=4,
    )
    result = graph.invoke({"questions": QUESTIONS})
    assert result["drafts"] == []
    assert seen(conn, fingerprint(insurance))
    conn.close()
