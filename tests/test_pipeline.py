import os.path
import numpy as np
from sklearn_pipeline_play.pipeline import PipelineWrapper
from sklearn_pipeline_play.pipeline import DataIngest
from sklearn_pipeline_play.pipeline import MainWrapper
from pytest import approx

def test_DataIngest():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    df = DataIngest(os.path.join(this_dir, 'data', 'simple_numbers.csv')).get()
    assert list(df.columns) == ['a', 'b', 'c'] # 3 dimensions
    assert len(df) == 9 # 9 samples

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

def test_MainWrapper():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_filename = os.path.join(this_dir, 'data', 'example.yaml')
    m = MainWrapper(['--yaml', str(yaml_filename)])
    cluster_distances = m.run()

    assert np.shape(cluster_distances) == (9, 3)
    assert cluster_distances[0] == approx([1.37402019, 2.26648292, 0.57924373])
    assert cluster_distances[-1] == approx([2.88632174, 3.32752544, 1.15848746])
