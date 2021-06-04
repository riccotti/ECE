import heapq
import numpy as np

from joblib import effective_n_jobs
from joblib import Parallel, delayed
from distutils.version import LooseVersion

from collections import defaultdict

from ece.ece import ECE
from ece.distr_ece import DistrECE
from ece.tree_ece import TreeECE
from ece.feature_ece import FeatureECE
from ece.random_ece import RandomECE
from ece.neighbor_ece import NeighborECE
from ece.cluster_ece import KMeansECE
from ece.casebased_ece import CaseBasedECE


"""If None (default), then draw X.shape[0] samples.
If int, then draw max_samples samples.
If float, then draw max_samples * X.shape[0] samples. Thus, max_samples should be in the interval (0, 1)."""

"""If int, then consider max_features features at each split.
If float, then max_features is a fraction and round(max_features * n_features) features are considered at each split.
If “auto”, then max_features=sqrt(n_features).
If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
If “log2”, then max_features=log2(n_features).
If None, then max_features=n_features."""


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs,
                                   dtype=np.int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


def _joblib_parallel_args(**kwargs):
    """Set joblib.Parallel arguments in a compatible way for 0.11 and 0.12+

    For joblib 0.11 this maps both ``prefer`` and ``require`` parameters to
    a specific ``backend``.

    Parameters
    ----------

    prefer : str in {'processes', 'threads'} or None
        Soft hint to choose the default backend if no specific backend
        was selected with the parallel_backend context manager.

    require : 'sharedmem' or None
        Hard condstraint to select the backend. If set to 'sharedmem',
        the selected backend will be single-host and thread-based even
        if the user asked for a non-thread based backend with
        parallel_backend.

    See joblib.Parallel documentation for more details
    """
    import joblib

    if joblib.__version__ >= LooseVersion('0.12'):
        return kwargs

    extra_args = set(kwargs.keys()).difference({'prefer', 'require'})
    if extra_args:
        raise NotImplementedError('unhandled arguments %s with joblib %s'
                                  % (list(extra_args), joblib.__version__))
    args = {}
    if 'prefer' in kwargs:
        prefer = kwargs['prefer']
        if prefer not in ['threads', 'processes', None]:
            raise ValueError('prefer=%s is not supported' % prefer)
        args['backend'] = {'threads': 'threading',
                           'processes': 'multiprocessing',
                           None: None}[prefer]

    if 'require' in kwargs:
        require = kwargs['require']
        if require not in [None, 'sharedmem']:
            raise ValueError('require=%s is not supported' % require)
        if require == 'sharedmem':
            args['backend'] = 'threading'
    return args


def _parallel_fit(estimator, b, X, est_idx, n_estimators, verbose):
    """
        Private function used to fit a single estimator in parallel."""
    if verbose > 0:
        print("building estimator %d of %d" % (est_idx + 1, n_estimators))

    estimator.fit(b, X)
    return estimator


def _parallel_get_counterfactuals(estimator, x, k, y_desiderd, constrain_into_ranges, search_diversity,
                                  est_idx, n_estimators, verbose):
    """
            Private function used to fit a single estimator in parallel."""
    if verbose > 0:
        print("searching cf %d of %d %s" % (est_idx + 1, n_estimators, type(estimator)))

    cf_list = estimator.get_counterfactuals(x, k, y_desiderd, constrain_into_ranges, search_diversity)

    if verbose > 0:
        print("ended searching cf %d of %d %s" % (est_idx + 1, n_estimators, type(estimator)))

    return cf_list


def _parallel_get_prototipes(estimator, x, k, beta, constrain_into_ranges, search_diversity,
                             est_idx, n_estimators, verbose):
    """
            Private function used to fit a single estimator in parallel."""
    if verbose > 0:
        print("searching cf %d of %d" % (est_idx + 1, n_estimators))

    pr_list = estimator.get_prototypes(x, k, beta, constrain_into_ranges, search_diversity)
    return pr_list


class EnsembleECE(ECE):

    def __init__(self, variable_features=None, weights=None, metric='euclidean', feature_names=None,
                 continuous_features=None, categorical_features_lists=None, normalize=True, pooler=None, tol=0.01,
                 base_estimator=None, n_estimators=100, max_samples=0.2, max_features='auto', min_features=2,
                 estimators_params=None, n_jobs=-1, verbose=0
                 ):

        super().__init__(variable_features, weights, metric, feature_names,
                         continuous_features, categorical_features_lists, normalize, pooler, tol)

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.min_features = min_features
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.categorical_features_lists = categorical_features_lists

        self.n_samples = None
        self.n_samples_bootstrap = None
        self.n_variable_features = None
        self.sample_indices_list = None
        self.sample_features_list = None
        self.estimators_ = None
        self.estimators_params = estimators_params
        self.cf_count = None

    def _get_base_estimator(self, variable_features):
        if self.base_estimator is None:
            base_estimator_ = DistrECE(variable_features, self.weights, self.metric, self.feature_names,
                                       self.continuous_features, self.categorical_features_lists,
                                       self.normalize, self.pooler,
                                       n_attempts=10, n_batch=1000, stopping_eps=0.01,
                                       kind='gaussian_matched')
        elif self.base_estimator == 'dist':
            base_estimator_ = DistrECE(variable_features, self.weights, self.metric, self.feature_names,
                                       self.continuous_features, self.categorical_features_lists,
                                       self.normalize, self.pooler, self.tol,
                                       n_attempts=10, n_batch=1000, stopping_eps=0.01,
                                       kind='gaussian_matched')
        elif self.base_estimator == 'tree':
            base_estimator_ = TreeECE(variable_features, self.weights, self.metric, self.feature_names,
                                      self.continuous_features, self.categorical_features_lists,
                                      self.normalize, self.pooler,
                                      use_instance_weights=False, kernel_width=None, min_samples_leaf=0.01,
                                      max_depth=None, closest_in_leaf=True)
        elif self.base_estimator == 'feat':
            base_estimator_ = FeatureECE(variable_features, self.weights, self.metric, self.feature_names,
                                         self.continuous_features, self.categorical_features_lists,
                                         self.normalize, self.pooler, self.tol,
                                         nbr_intervals=10, nbr_features_to_test=1)
        elif self.base_estimator == 'neig':
            base_estimator_ = NeighborECE(variable_features, self.weights, self.metric, self.feature_names,
                                          self.continuous_features, self.categorical_features_lists,
                                          self.normalize, self.pooler,
                                          random_samples=100)
        elif self.base_estimator == 'clus':
            base_estimator_ = KMeansECE(variable_features, self.weights, self.metric, self.feature_names,
                                        self.continuous_features, self.categorical_features_lists,
                                        self.normalize, self.pooler,
                                        n_clusters=10,
                                        init='k-means++',
                                        n_init=10,
                                        max_iter=300,
                                        tol=0.0001)
        elif self.base_estimator == 'cb':
            base_estimator_ = CaseBasedECE(variable_features, self.weights, self.metric, self.feature_names,
                                           self.continuous_features, self.categorical_features_lists,
                                           self.normalize, self.pooler,
                                           random_samples=None,
                                           diff_features=2,
                                           tolerance=0.001)
        elif self.base_estimator == 'rand':
            base_estimator_ = RandomECE(variable_features, self.weights, self.metric, self.feature_names,
                                        self.continuous_features, self.categorical_features_lists,
                                        self.normalize, self.pooler,
                                        n_attempts=100,
                                        n_max_attempts=1000,
                                        proba=0.5)
        elif self.base_estimator == 'ens':
            base_estimator_ = None
        else:
            raise Exception('Unsupported base_estimator %s' % self.base_estimator)
        return base_estimator_

    def _get_estimator_param(self, variable_features):

        if self.cf_count is None:
            self.cf_count = {k: 0 for k in self.estimators_params}

        min_counts = np.min(list(self.cf_count.values()))
        self.sample_from = np.array([k for k, v in self.cf_count.items() if v == min_counts])
        cf_type = np.random.choice(self.sample_from)
        self.cf_count[cf_type] += 1

        params = self.estimators_params[cf_type]

        if cf_type == 'rand':
            base_estimator_ = RandomECE(variable_features, self.weights, self.metric, self.feature_names,
                                        self.continuous_features, self.categorical_features_lists,
                                        self.normalize, self.pooler,
                                        n_attempts=params.get('n_attempts', 100),
                                        n_max_attempts=params.get('n_max_attempts', 1000),
                                        proba=params.get('proba', 0.5))

        elif cf_type == 'feat':
            base_estimator_ = FeatureECE(variable_features, self.weights, self.metric, self.feature_names,
                                         self.continuous_features, self.categorical_features_lists,
                                         self.normalize, self.pooler,
                                         nbr_intervals=params.get('nbr_intervals', 10),
                                         nbr_features_to_test=params.get('nbr_features_to_test', 1),
                                         tol=params.get('tol', 0.01))
        elif cf_type == 'neig':
            base_estimator_ = NeighborECE(variable_features, self.weights, self.metric, self.feature_names,
                                          self.continuous_features, self.categorical_features_lists,
                                          self.normalize, self.pooler,
                                          random_samples=params.get('random_samples', 100))
        elif cf_type == 'clus':
            base_estimator_ = KMeansECE(variable_features, self.weights, self.metric, self.feature_names,
                                        self.continuous_features, self.categorical_features_lists,
                                        self.normalize, self.pooler,
                                        n_clusters=params.get('n_clusters', 10),
                                        init=params.get('init', 'k-means++'),
                                        n_init=params.get('n_init', 10),
                                        max_iter=params.get('max_iter', 300),
                                        tol=params.get('tol', 0.0001))
        elif cf_type == 'tree':
            base_estimator_ = TreeECE(variable_features, self.weights, self.metric, self.feature_names,
                                      self.continuous_features, self.categorical_features_lists,
                                      self.normalize, self.pooler,
                                      use_instance_weights=params.get('use_instance_weights', False),
                                      kernel_width=params.get('kernel_width', None),
                                      min_samples_leaf=params.get('min_samples_leaf', 0.01),
                                      max_depth=params.get('max_depth', None),
                                      closest_in_leaf=params.get('closest_in_leaf', True))
        elif cf_type == 'cb':
            base_estimator_ = CaseBasedECE(variable_features, self.weights, self.metric, self.feature_names,
                                           self.continuous_features, self.categorical_features_lists,
                                           self.normalize, self.pooler,
                                           random_samples=params.get('random_samples', None),
                                           diff_features=params.get('diff_features', 2),
                                           tolerance=params.get('tolerance', 0.001))

        elif cf_type == 'dist':
            base_estimator_ = DistrECE(variable_features, self.weights, self.metric, self.feature_names,
                                       self.continuous_features, self.categorical_features_lists,
                                       self.normalize, self.pooler,
                                       n_attempts=params.get('n_attempts', 10),
                                       n_batch=params.get('n_batch', 1000),
                                       stopping_eps=params.get('stopping_eps', 0.01),
                                       kind=params.get('kind', 'gaussian_matched'),
                                       tol=params.get('tol', 0.01)
                                       )

        elif cf_type == 'ens':
            base_estimator_ = EnsembleECE(variable_features, self.weights, self.metric, self.feature_names,
                                          self.continuous_features, self.categorical_features_lists,
                                          self.normalize, self.pooler,
                                          base_estimator=params.get('base_estimator', None),
                                          n_estimators=params.get('n_estimators', 10),
                                          max_samples=params.get('max_samples', 0.2),
                                          max_features=params.get('max_features', 'auto'),
                                          n_jobs=params.get('n_jobs', -1),
                                          verbose=params.get('verbose', 0))
        else:
            raise Exception('Unsupported base_estimator %s' % cf_type)

        return base_estimator_

    def fit(self, b, X):
        super().fit(b, X)

        self.n_samples = X.shape[0]
        self.n_variable_features = len(self.variable_features)

        if self.max_samples is None:
            self.n_samples_bootstrap = self.n_samples
        elif isinstance(self.max_samples, int):
            self.n_samples_bootstrap = self.max_samples
        elif isinstance(self.max_samples, float):
            self.max_samples = self.max_samples
            if self.max_samples > 1:
                self.max_samples /= 100.0
            self.n_samples_bootstrap = int(self.n_samples * self.max_samples)
        else:
            raise Exception('Unsupported max_samples %s' % self.max_samples)

        if self.max_features is None:
            self.max_features = self.n_variable_features
        elif isinstance(self.max_features, int):
            self.max_features = self.max_features
        elif isinstance(self.max_features, float):
            self.max_features = int(self.n_variable_features * self.max_features)
        elif self.max_features == 'auto':
            self.max_features = int(np.sqrt(self.n_variable_features))
        elif self.max_features == 'sqrt':
            self.max_features = int(np.sqrt(self.n_variable_features))
        elif self.max_features == 'log2':
            self.max_features = int(np.log2(self.n_variable_features))
        else:
            raise Exception('Unsupported max_features %s' % self.max_features)
        self.max_features = max(self.min_features, self.max_features)

        # print(self.n_samples, self.n_samples_bootstrap, self.max_features)
        self.sample_indices_list, self.sample_features_list = self.generate_boosting_samples(
            self.n_samples, self.n_samples_bootstrap, self.max_features)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        estimators = list()
        if self.estimators_params is None:
            for sfl in self.sample_features_list:
                estimator = self._get_base_estimator(sfl)
                estimators.append(estimator)
        else:
            for sfl in self.sample_features_list:
                estimator = self._get_estimator_param(sfl)
                estimators.append(estimator)

        estimators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, **_joblib_parallel_args(prefer='threads'))(
            delayed(_parallel_fit)(est, b, X[sil, :], est_idx, self.n_estimators, self.verbose)
            for est_idx, est, sil in zip(np.arange(self.n_estimators), estimators, self.sample_indices_list))

        self.estimators_ = estimators

    def generate_boosting_samples(self, n_samples, n_samples_bootstrap, max_features):

        sample_indices_list = list()
        sample_features_list = list()

        unsampled_indices = np.arange(n_samples)
        unsampled_features = self.variable_features[::]
        # unsampled_features = np.arange(self.n_features)

        sample_counts = np.zeros(n_samples)
        # features_counts = np.zeros(n_features)
        features_counts = {k: 0 for k in unsampled_features}

        indices_range = np.arange(n_samples)
        # features_range = np.arange(n_features)

        for i in range(self.n_estimators):
            sample_indices = np.random.choice(unsampled_indices, n_samples_bootstrap, replace=True)
            sample_indices_list.append(sample_indices)
            sample_counts += np.bincount(sample_indices, minlength=n_samples)
            unsampled_indices = indices_range[sample_counts == 0]
            if len(unsampled_indices) < 2:
                unsampled_indices = np.arange(n_samples)

            sample_features = sorted(np.random.choice(unsampled_features, max_features, replace=False))
            sample_features_list.append(sample_features)
            # print(i, features_counts, sample_features, n_features, np.bincount(sample_features, minlength=n_features))
            # features_counts += np.bincount(sample_features, minlength=n_features)
            # unsampled_features = features_range[features_counts == 0]
            for k in sample_features:
                features_counts[k] += 1
            unsampled_features = np.array([k for k, v in features_counts.items() if v == 0])
            if len(unsampled_features) < max_features:
                unsampled_features = self.variable_features[::]
                # unsampled_features = np.arange(n_features)

        sample_indices_list = np.array(sample_indices_list)
        sample_features_list = np.array(sample_features_list)

        return sample_indices_list, sample_features_list

    # heuristic
    def heuristic_kcover(self, x, cf_list_all, k=5, cf_rate=0.5, cf_rate_incr=0.1):

        # x = x.reshape(1, -1)
        x = np.expand_dims(x, 0)
        nx = self.scaler.transform(x)

        ncf_list_all = self.scaler.transform(cf_list_all)
        dist_x_cf = self.cdist(nx, ncf_list_all, metric=self.metric, w=self.weights)
        dist_cf_cff = np.zeros(len(cf_list_all))

        cf_list = list()
        cf_idx_list = list()
        while len(cf_list) < k:
            if len(cf_list) == 0:
                dist = dist_x_cf
            else:
                dist = cf_rate * dist_cf_cff + (1 - cf_rate) * dist_x_cf
            idx = np.argmin(dist)
            cf_idx_list.append(idx)
            cf_list.append(cf_list_all[idx])
            dist_cf_cff = 1 / (1 + np.mean(self.cdist(ncf_list_all[cf_idx_list], ncf_list_all,
                                                      metric=self.metric, w=self.weights), axis=0))
            for cfi in cf_idx_list:
                dist_cf_cff[cfi] = np.inf
            cf_rate = cf_rate - cf_rate * cf_rate_incr

        cf_list = np.array(cf_list)

        return cf_list

    def selected_cf_distance(self, x, selected, lambda_par=1.0, knn_dist=False, knn_list=None, lconst=None):

        if not knn_dist:
            dist_ab = 0.0
            dist_ax = 0.0
            for i in range(len(selected)):
                a = np.expand_dims(selected[i], 0)
                for j in range(i + 1, len(selected)):
                    b = np.expand_dims(selected[j], 0)
                    dist_ab += self.cdist(a, b, metric=self.metric, w=self.weights)[0][0]
                dist_ax += self.cdist(a, x, metric=self.metric, w=self.weights)[0][0]

            coef_ab = 1 / (len(selected) * len(selected)) if len(selected) else 0.0
            coef_ax = lambda_par / len(selected) if len(selected) else 0.0

        else:
            dist_ax = 0.0
            common_cfs = set()
            for i in range(len(selected)):
                a = np.expand_dims(selected[i], 0)
                knn_a = knn_list[a.tobytes()]
                common_cfs |= knn_a
                dist_ax += self.cdist(a, x, metric=self.metric, w=self.weights)[0][0]
            dist_ab = len(common_cfs)

            coef_ab = 1.0
            coef_ax = 2.0 * lconst

        dist = coef_ax * dist_ax - coef_ab * dist_ab
        # dist = coef_ab * dist_ab - coef_ax * dist_ax
        return dist

    def get_best_cf(self, x, selected, cf_list_all, lambda_par=1.0, submodular=True,
                    knn_dist=False, knn_list=None, lconst=None):
        min_d = np.inf
        best_i = None
        best_d = None
        d_w_a = self.selected_cf_distance(x, selected, lambda_par, knn_dist, knn_list, lconst)
        for i, cf in enumerate(cf_list_all):
            d_p_a = self.selected_cf_distance(x, selected + [cf], lambda_par)
            d = d_p_a - d_w_a if submodular else d_p_a  # submudular -> versione derivata
            if d < min_d:
                best_i = i
                best_d = d_p_a
                min_d = d

        return best_i, best_d

    # greedy cover
    def greedy_kcover(self, x, cf_list_all, k=5, lambda_par=1.0, submodular=True, knn_dist=False):

        x = np.expand_dims(x, 0)
        nx = self.scaler.transform(x)

        ncf_list_all = self.scaler.transform(cf_list_all)

        lconst = None
        knn_list = None
        if knn_dist:
            dist_x_cf = self.cdist(nx, ncf_list_all, metric=self.metric, w=self.weights)
            d0 = np.argmin(dist_x_cf)
            lconst = 0.5 / (-d0) if d0 != 0.0 else 0.5

            # cf_dist_matrix = np.mean(self.cdist(ncf_list_all, ncf_list_all,
            #                                     metric=self.metric, w=self.weights), axis=0)
            cf_dist_matrix = self.cdist(ncf_list_all, ncf_list_all, metric=self.metric, w=self.weights)

            knn_list = dict()
            for idx, knn in enumerate(np.argsort(cf_dist_matrix, axis=1)[:, 1:k+1]):
                cf_core_key = np.expand_dims(cf_list_all[idx], 0).tobytes()
                knn_set = set([np.expand_dims(cf_list_all[nn], 0).tobytes() for nn in knn])
                knn_list[cf_core_key] = knn_set

        cf_list = list()
        cf_selected = list()
        ncf_selected = list()
        min_dist = np.inf
        while len(ncf_selected) < k:
            idx, dist = self.get_best_cf(nx, ncf_selected, ncf_list_all, lambda_par, submodular,
                                         knn_dist, knn_list, lconst)
            cf_selected.append(self.scaler.inverse_transform(ncf_list_all[idx]))
            ncf_selected.append(ncf_list_all[idx])
            ncf_list_all = np.delete(ncf_list_all, idx, axis=0)
            if dist < min_dist:
                min_dist = dist
                cf_list = cf_selected

        cf_list = np.array(cf_list)

        return cf_list

    # greedy accelerated cover
    def greedy_accelerated_kcover(self, x, cf_list_all, k=5, lambda_par=1.0, submodular=True, knn_dist=False):

        x = np.expand_dims(x, 0)
        nx = self.scaler.transform(x)

        ncf_list_all = self.scaler.transform(cf_list_all)

        lconst = None
        knn_list = None
        if knn_dist:
            dist_x_cf = self.cdist(nx, ncf_list_all, metric=self.metric, w=self.weights)
            d0 = np.argmin(dist_x_cf)
            lconst = 0.5 / (-d0) if d0 != 0.0 else 0.5

            # cf_dist_matrix = np.mean(squareform(self.cdist(ncf_list_all, ncf_list_all,
            #                                     metric=self.metric, w=self.weights)), axis=0)
            cf_dist_matrix = self.cdist(ncf_list_all, ncf_list_all, metric=self.metric, w=self.weights)

            knn_list = dict()
            for idx, knn in enumerate(np.argsort(cf_dist_matrix, axis=1)[:, 1:k]):
                cf_core_key = np.expand_dims(cf_list_all[idx], 0).tobytes()
                knn_set = set([np.expand_dims(cf_list_all[nn], 0).tobytes() for nn in knn])
                knn_list[cf_core_key] = knn_set

        ncf_selected = list()
        pset = list()
        for idx, cf in enumerate(ncf_list_all):
            _, dist = self.get_best_cf(nx, [], ncf_list_all, lambda_par, submodular, knn_dist, knn_list, lconst)
            pset.append([dist, list(cf), idx])

        heapq.heapify(pset)

        for i in range(k):
            best = heapq.heappop(pset)
            _, best[0] = self.get_best_cf(nx, ncf_selected + [best[1]], ncf_list_all,
                                          lambda_par, submodular, knn_dist, knn_list, lconst)
            newbest = heapq.heappushpop(pset, best)
            while newbest[0] < best[0]:
                best = newbest
                _, best[0] = self.get_best_cf(nx, ncf_selected + [best[1]], ncf_list_all,
                                              lambda_par, submodular, knn_dist, knn_list, lconst)
                newbest = heapq.heappushpop(pset, best)
            if best[0] <= 0:
                break
            ncf_selected.append(ncf_list_all[best[2]])

        cf_list = np.array(self.scaler.inverse_transform(ncf_selected))

        return cf_list

    def majority_kcover(self, cf_list_all, k=5):
        cf_count = defaultdict(int)
        cf_map = dict()
        # for cf_list in cf_list_all:
        #     for cf in cf_list:
        #         cf_count[cf.tobytes()] += 1
        #         cf_map[cf.tobytes()] = cf
        for cf in cf_list_all:
            # for cf in cf_list:
            cf_count[cf.tobytes()] += 1
            cf_map[cf.tobytes()] = cf

        cf_key_list = sorted(cf_count, key=cf_count.get, reverse=True)[:k]
        cf_list = np.array([np.array(cf_map[cfk]) for cfk in cf_key_list])
        return cf_list

    def get_counterfactuals(self, x, k=5, y_desiderd=None, constrain_into_ranges=True, search_diversity=False,
                            covertype='naive', lambda_par=1.0, cf_rate=0.5, cf_rate_incr=0.1, return_all=False,
                            cf_list_all=None
                            ):

        if cf_list_all is None:
            # Assign chunk of trees to jobs
            n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

            cf_list_all = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, **_joblib_parallel_args(prefer='threads'))(
                delayed(_parallel_get_counterfactuals)(est, x, k, y_desiderd, constrain_into_ranges, search_diversity,
                                                       est_idx, self.n_estimators, self.verbose)
                for est_idx, est, sil in zip(np.arange(self.n_estimators), self.estimators_, self.sample_indices_list))

            # print('quiiii')
            if self.pooler:
                cf_list_all_tmp = list()
                for cf_list in cf_list_all:
                    if len(cf_list) > 0:
                        # print(self.b.predict(cf_list))
                        cf_list_tmp = self.pooler.transform(cf_list)
                        cf_list_all_tmp.append(cf_list_tmp)
                cf_list_all = cf_list_all_tmp
                x = self.pooler.transform(np.array([x]))[0]

            if covertype == 'majority':
                if len(cf_list_all) > 0:
                    cflist_tmp = [np.array(cf) for cf in cf_list_all if len(cf) > 0]
                    cf_list_all = np.vstack(cflist_tmp) if len(cflist_tmp) > 0 else np.array([])
                else:
                    cf_list_all = np.array([])
            else:
                cf_list_all = set([tuple(cf) for cf_list in cf_list_all for cf in cf_list])

            cf_list_all = np.array([np.array(cf) for cf in cf_list_all])

        else:
            if covertype != 'majority':
                cf_list_all = set([tuple(cf) for cf in cf_list_all])
                cf_list_all = np.array([np.array(cf) for cf in cf_list_all])

            if self.pooler:
                cf_list_all_tmp = list()
                for cf_list in cf_list_all:
                    cf_list_tmp = self.pooler.transform(cf_list)
                    cf_list_all_tmp.append(cf_list_tmp)
                cf_list_all = cf_list_all_tmp
                x = self.pooler.transform(np.array([x]))[0]

        # x = self.scaler.transform(x)
        # cf_list_all = self.scaler.transform(cf_list_all)

        if len(cf_list_all) > k:
            if covertype == 'naive':
                cf_list = self.greedy_kcover(x, cf_list_all, k, lambda_par, submodular=False, knn_dist=False)
            elif covertype == 'naive-sub':
                cf_list = self.greedy_kcover(x, cf_list_all, k, lambda_par, submodular=True, knn_dist=False)
            elif covertype == 'knn':
                cf_list = self.greedy_kcover(x, cf_list_all, k, lambda_par, submodular=False, knn_dist=True)
            elif covertype == 'knn-sub':
                cf_list = self.greedy_kcover(x, cf_list_all, k, lambda_par, submodular=True, knn_dist=True)
            elif covertype == 'knn-acc':
                cf_list = self.greedy_accelerated_kcover(x, cf_list_all, k, lambda_par, submodular=False, knn_dist=True)
            elif covertype == 'knn-acc-sub':
                cf_list = self.greedy_accelerated_kcover(x, cf_list_all, k, lambda_par, submodular=True, knn_dist=True)
            elif covertype == 'heuristic':
                cf_list = self.heuristic_kcover(x, cf_list_all, k, cf_rate, cf_rate_incr)
            elif covertype == 'majority':
                cf_list = self.majority_kcover(cf_list_all, k)
            else:
                raise Exception('Unknown cover type %s' % covertype)

            cf_list = cf_list if not self.pooler else self.pooler.inverse_transform(cf_list)
            cf_list_all = cf_list_all if not self.pooler else self.pooler.inverse_transform(cf_list_all)

            if return_all:
                return cf_list, cf_list_all

            return cf_list

        if return_all:
            return cf_list_all, cf_list_all

        if len(cf_list_all) > 0:
            cf_list_all = cf_list_all if not self.pooler else self.pooler.inverse_transform(cf_list_all)

        return cf_list_all

    def get_prototypes(self, x, k=5, beta=0.5, constrain_into_ranges=True, search_diversity=False,
                       covertype='greedy', lambda_par=1.0, cf_rate=0.5, cf_rate_incr=0.1):

        pr_list_all = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, **_joblib_parallel_args(prefer='threads'))(
            delayed(_parallel_get_prototipes)(est, x, k, beta, constrain_into_ranges, search_diversity,
                                              est_idx, self.n_estimators, self.verbose)
            for est_idx, est, sil in zip(np.arange(self.n_estimators), self.estimators_, self.sample_indices_list))

        if covertype != 'majority':
            pr_list_all = set([tuple(pr) for pr_list in pr_list_all for pr in pr_list])

        pr_list_all = np.array([np.array(pr) for pr in pr_list_all])

        if len(pr_list_all) > k:
            if covertype == 'greedy':
                pr_list = self.greedy_kcover(x, pr_list_all, k, lambda_par)
            elif covertype == 'heuristic':
                pr_list = self.heuristic_kcover(x, pr_list_all, k, cf_rate, cf_rate_incr)
            elif covertype == 'majority':
                pr_list = self.majority_kcover(pr_list_all, k)
            else:
                raise Exception('Unknown cover type %s' % covertype)
            return pr_list

        return pr_list_all
