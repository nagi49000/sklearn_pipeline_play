import os.path
import numpy as np
from sklearn_pipeline_play.pipeline import PipelineWrapper
from sklearn_pipeline_play.pipeline import DataIngest
from sklearn_pipeline_play.pipeline import MainWrapper
from sklearn_pipeline_play.pipeline import NoiseAdder
from pytest import approx


def test_DataIngest():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    df = DataIngest(os.path.join(this_dir, 'data', 'simple_numbers.csv')).get()
    assert list(df.columns) == ['a', 'b', 'c']  # 3 dimensions
    assert len(df) == 9  # 9 samples


def test_PipelineWrapper():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    df = DataIngest(os.path.join(this_dir, 'data', 'simple_numbers.csv')).get()
    pw = PipelineWrapper({})
    cluster_distances = pw.fit_transform(df)
    # 9 samples. For each sample, 8 numbers representing the distance from each cluster centre
    assert np.shape(cluster_distances) == (9, 8)

    yaml_dict = {'cluster': {'n_clusters': 3,
                             'random_state': 0},
                 'dim_reduce': {'random_state': 0}}
    pw = PipelineWrapper(yaml_dict)
    cluster_distances = pw.fit_transform(df)
    assert np.shape(cluster_distances) == (9, 3)
    assert cluster_distances[0] == approx([1.37402019, 2.26648292, 0.57924373])
    assert cluster_distances[-1] == approx([2.88632174, 3.32752544, 1.15848746])

    yaml_dict = {'cluster': {'n_clusters': 3,
                             'random_state': 0},
                 'dim_reduce': {'random_state': 0},
                 'add_noise': {'random_state': 0,
                               'loc': 10.0,
                               'scale': 0.1}
                 }
    pw = PipelineWrapper(yaml_dict)
    cluster_distances = pw.fit_transform(df)
    assert np.shape(cluster_distances) == (9, 3)
    assert cluster_distances[0] == approx([2.2093906, 0.94026417, 1.89990178])
    assert cluster_distances[-1] == approx([3.5363537, 2.63086021, 0.0])


def test_MainWrapper():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_filename = os.path.join(this_dir, 'data', 'example.yaml')
    m = MainWrapper(['--yaml', str(yaml_filename)])
    cluster_distances = m.run()

    assert np.shape(cluster_distances) == (9, 3)
    assert cluster_distances[0] == approx([1.37402019, 2.26648292, 0.57924373])
    assert cluster_distances[-1] == approx([2.88632174, 3.32752544, 1.15848746])


def test_NoiseAdder():
    n = NoiseAdder()
    X = n.transform(np.array([[0, 0, 0], [0, 0, 0]]))
    assert X.shape == (2, 3)
    assert X[0] == approx([0.0, 0.0, 0.0])
    assert X[1] == approx([0.0, 0.0, 0.0])

    n = NoiseAdder(random_state=0, loc=1, scale=0.1)
    X = n.transform(np.array([[10, 20, 30], [40, 50, 60]]))
    assert X.shape == (2, 3)
    assert X[0] == approx([11.17640523, 21.04001572, 31.0978738])
    assert X[1] == approx([41.22408932, 51.1867558, 60.90227221])

    n = n.fit([])  # should be trivial return
    X = n.transform(np.array([[10, 20, 30], [40, 50, 60]]))
    assert X.shape == (2, 3)
    assert X[0] == approx([11.17640523, 21.04001572, 31.0978738])
    assert X[1] == approx([41.22408932, 51.1867558, 60.90227221])
