import random
from faker import Faker
from faker.providers import internet
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class FakeDataGenerator():
    def get_domain_names(self, number_of_records, levels=[2], seed=None):
        if seed is not None:
            random.seed(seed)
            Faker.seed(seed)
        fake = Faker()
        fake.add_provider(internet)
        for _ in range(number_of_records):
            yield fake.hostname(levels=random.choice(levels))
