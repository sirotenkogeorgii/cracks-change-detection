import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings
import os


class CrackSampler:
    def __init__(
        self,
        y_transletion_file_path: str | os.PathLike,
        x_transletion_file_path: str | os.PathLike,
        scale_file_path: str | os.PathLike, 
        rotation_file_path: str | os.PathLike, 
        dist_names: list[str] = None
    ) -> None:

        self.fluences = None
        self.fluence2index = None

        self.y_transletion_data = self._process_file(y_transletion_file_path)
        self.x_transletion_data = self._process_file(x_transletion_file_path)
        self.scale_data = self._process_file(scale_file_path)
        self.rotation_data = self._process_file(rotation_file_path)

        self.fluence_probabilities = [1 / self.fluences.size for _ in range(self.fluences.size)]
        if dist_names is None:
            dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme", "uniform", 
                    "alpha", "beta", "expon", "exponnorm", "gennorm", "gamma", "cauchy", "cosine", "gennorm", 
                    "genexpon", "genpareto", "gausshyper", "gengamma", "genhyperbolic", "genlogistic"
                    ]
        self.dist_names = dist_names
        

    def _process_file(self, file_path: str | os.PathLike) -> np.ndarray:
        df = pd.read_csv(file_path)

        # cells with Nan values in the DataFrame.
        number_of_nans = df.isnull().sum().sum()
        if number_of_nans > 0:
            raise Exception(f"The file is not in the correct format. File: {file_path}. Missing values: {number_of_nans}.")

        if self.fluences is None:
            self.fluences = np.array((list(map(float, list(df.columns)))), dtype=np.double)
            self.fluence2index = {str(f) : i for i, f in enumerate(self.fluences)}
        
        return df.to_numpy()


    def _get_best_distribution(self, data: np.ndarray) -> tuple[str, list[float]]:
        dist_results = []
        params = {}
        for dist_name in self.dist_names:
            dist = getattr(stats, dist_name)
            param = dist.fit(data)

            params[dist_name] = param
            _, p = stats.kstest(data, dist_name, args=param)
            dist_results.append((dist_name, p))

        best_dist_name, _ = (max(dist_results, key=lambda item: item[1]))

        return best_dist_name, params[best_dist_name]


    def _sample_cracks(self, cracks_num: int, fluence: int = None) -> list[tuple[float, float, float, float]]:
        warnings.simplefilter("ignore")
        if fluence is None:
            fluence = self.sample_fluence()

        result = []    
        sampled_index = self.fluence2index[str(fluence)]
        for parameter in [self.y_transletion_data, self.x_transletion_data, self.scale_data, self.rotation_data]:
            column = parameter[:, sampled_index]
            ditribution_name, ditribution_params = self._get_best_distribution(column)
            distribution = getattr(stats, ditribution_name)
            samples = distribution.rvs(*ditribution_params, size=cracks_num)
            result.append(samples)

        # returns a list of tuples (y_transletion, x_transletion, scale, rotation)
        return list(zip(*result))


    def sample_fluence(self) -> int:
        return np.random.choice(self.fluences, 1, self.fluence_probabilities)[0]


    def sample_cracks(self, cracks_num: int, fluence: int = None) -> list[tuple[float, float, float, float]]:
        with warnings.catch_warnings():  
            warnings.simplefilter("ignore")
            return self._sample_cracks(cracks_num, fluence)