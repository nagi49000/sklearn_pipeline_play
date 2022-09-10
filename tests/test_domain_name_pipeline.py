import types
from sklearn_pipeline_play.domain_name_pipeline import(
    FakeDataGenerator,
    Tokenizer,
    StopWordRemover,
    get_references,
    ReferenceAdder
)

def test_fake_data_generator():
    g = FakeDataGenerator().get_hostnames(3, seed=42)
    assert isinstance(g, types.GeneratorType)
    assert list(g) == [
        "email-10.hill.yang.com",
        "db-13.wagner.gonzalez.com",
        "laptop-94.blake.henderson.com",
    ]
    g = FakeDataGenerator().get_hostnames(5, levels=[2,3,4], seed=42)
    assert list(g) == [
        "email-10.hill.yang.johnson.wagner.net",
        "srv-86.cole.blake.biz",
        "desktop-42.bernard.davis.com",
        "web-59.roman-blair.martinez.calderon-montgomery.ray-bush.com",
        "email-34.wyatt.ramirez-reid.martin-kelly.com",
    ]


def test_tokenizer():
    g = FakeDataGenerator().get_hostnames(3, seed=42)
    t = Tokenizer().fit(g).transform(g)
    assert isinstance(t, types.GeneratorType)
    assert list(t) == [
        ["email-10", "hill", "yang", "com"],
        ["db-13", "wagner", "gonzalez", "com"],
        ["laptop-94", "blake", "henderson", "com"],
    ]
    t = Tokenizer().transform(["..foo...bar..."])
    assert list(t) == [["foo", "bar"]]


def test_stop_word_remover():
    tokens = [
        ["alpha", "bravo", "foo", "com"],
        ["bar", "charlie", "delta", "net"]
    ]
    t = StopWordRemover({"foo", "bar"}).fit(tokens).transform(tokens)
    assert isinstance(t, types.GeneratorType)
    assert list(t) == [
        ["alpha", "bravo", "com"],
        ["charlie", "delta", "net"]
    ]


def test_reference_identifier():
    assert get_references([]) == []
    assert get_references(["bob", "com"]) == ["country"]
    assert get_references(["bob", "edu"]) == ["country", "uni"]


def test_reference_adder():
    r = ReferenceAdder(get_references)
    tokens = [
        [],
        ["bob", "com"],
        ["bob", "edu"]
    ]
    with_refs = r.fit(tokens).transform(tokens)
    assert list(with_refs) == [
        {'tokens': [], 'references': []},
        {'tokens': ['bob', 'com'], 'references': ['country']},
        {'tokens': ['bob', 'edu'], 'references': ['country', 'uni']}
    ]
