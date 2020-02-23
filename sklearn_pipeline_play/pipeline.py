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
    """ Wraps a scikit learn Pipeline. Uses a nested dictionary of parameters to configure the pipeline """
    def __init__(self, yaml_dict):
        """ yaml_dict - dict - nested dictionary of (1st level) the steps in the pipeline, and (2nd level) step params """
        self._params = self._get_params_from_yaml_dict(yaml_dict)

    def _get_params_from_yaml_dict(self, yd):
        """ yd - dict - nested dictionary of (1st level) the steps in the pipeline, and (2nd level) step params
            returns a dictionary of params of the pipeline in scikit learn Pipeline lingo
        """
        params = {}
        for step in {'normalise', 'dim_reduce', 'cluster'} & set(yd.keys()):
            params.update({step+'__'+k:v for k,v in yd[step].items()})
        return params

    def _get_pipeline(self, params_dict):
        """ params_dict - dict - params of the pipeline in scikit learn Pipeline lingo
            returns a configured pipeline
        """
        p = Pipeline(steps=[('normalise', StandardScaler()),
                            ('dim_reduce', PCA()),
                            ('cluster', KMeans())])
        p.set_params(**params_dict)
        return p

    def fit_transform(self, data):
        """ data - iterable - data source where zeroth index goes over samples, 1st index over features
            returns the result of running fit_transform over the pipeline
        """
        p = self._get_pipeline(self._params)
        return p.fit_transform(data)

class DataIngest:
    """ Wrapper for ingesting a table of data """
    def __init__(self, source):
        """ source - str - specifies some kind of source. Currently, only csv supported """
        self._source = source

    def get(self):
        """ gets the source data as a pandas DataFrame """
        return pd.read_csv(self._source)

class MainWrapper:
    """ wrapper for calling Pipeline from the command line """
    def __init__(self, argv):
        """ argv - list<str> - as hoovered up from sys.argv """
        self._argv = argv

    def _parse_args(self, cmd_line_list):
        """ cmd_line_list - list<str> - as hoovered up from sys.argv
            returns a dictionary of arguments parsed at the command line
        """
        parser = ArgumentParser()
        parser.add_argument('--yaml', help='yaml file specifying config to run')
        args = parser.parse_args(cmd_line_list)
        return vars(args)

    def run(self):
        """ runs from command line args through to pipeline. Returns pipeline results """
        args = self._parse_args(self._argv)
        with open(args['yaml']) as yaml_file:
            yaml_dict = yaml.safe_load(yaml_file) # returns list<dict>
        yaml_dict = yaml_dict[0]['pipeline']
        data = DataIngest(yaml_dict['data']).get()
        return PipelineWrapper(yaml_dict).fit_transform(data)

if __name__ == '__main__':
    print(MainWrapper(sys.argv).run())
