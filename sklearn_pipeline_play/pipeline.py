import functools
import pandas as pd
from argparse import ArgumentParser
import yaml
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class PipelineWrapper:
    def __init__(self, yaml_dict):
        self._params = self._get_params_from_args(yaml_dict)

    def _get_params_from_args(self, args):
        params = {}
        for step in {'normalise', 'dim_reduce', 'cluster'} & set(args.keys()):
            params.update({step+'__'+k:v for k,v in args[step].items()})
        return params

    def _get_pipeline(self, params_dict):
        p = Pipeline(steps=[('normalise', StandardScaler()),
                            ('dim_reduce', PCA()),
                            ('cluster', KMeans())])
        p.set_params(**params_dict)
        return p

    def fit_transform(self, data):
        p = self._get_pipeline(self._params)
        return p.fit_transform(data)

class DataIngest:
    def __init__(self, source):
        self._source = source

    def get(self):
        return pd.read_csv(self._source)

class MainWrapper:
    def __init__(self, argv):
        self._argv = argv

    def _parse_args(self, cmd_line_list):
        parser = ArgumentParser()
        parser.add_argument('--yaml', help='yaml file specifying config to run')
        args = parser.parse_args(cmd_line_list)
        return vars(args)

    def run(self):
        args = self._parse_args(self._argv)
        with open(args['yaml']) as yaml_file:
            yaml_dict = yaml.safe_load(yaml_file)
        yaml_dict = yaml_dict[0]['pipeline']
        data = DataIngest(yaml_dict['data']).get()
        return PipelineWrapper(yaml_dict).fit_transform(data)

if __name__ == '__main__':
    print(MainWrapper(sys.argv).run())
