import types
from sklearn_pipeline_play.domain_name_pipeline import FakeDataGenerator


def test_fake_data_generator():
    g = FakeDataGenerator().get_domain_names(3, seed=42)
    assert isinstance(g, types.GeneratorType)
    assert list(g) == [
        "email-10.hill.yang.com",
        "db-13.wagner.gonzalez.com",
        "laptop-94.blake.henderson.com",
    ]
