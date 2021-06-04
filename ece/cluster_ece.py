import numpy as np

from sklearn.cluster import KMeans

from ece.ece import ECE


class KMeansECE(ECE):

    def __init__(self, variable_features=None, weights=None, metric='euclidean', feature_names=None,
                 continuous_features=None, categorical_features_lists=None, normalize=True, pooler=None,
                 n_clusters=10, init='k-means++', n_init=10, max_iter=300, tol=0.0001):
        super().__init__(variable_features, weights, metric, feature_names, continuous_features,
                         categorical_features_lists, normalize, pooler)

        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol

        self.kmeans = None
        self.n_class_centroids = None

    def fit(self, b, X):
        super().fit(b, X)
        self.kmeans = KMeans(n_clusters=self.n_clusters, init=self.init, n_init=self.n_init,
                             max_iter=self.max_iter, tol=self.tol)
        classes = np.unique(self.y)
        self.n_class_centroids = dict()
        for y_val in classes:
            X_cfc = self.X[self.y != y_val]
            nX_cfc = self.scaler.transform(X_cfc)
            self.kmeans.fit(nX_cfc[:, self.variable_features])
            self.n_class_centroids[y_val] = self.kmeans.cluster_centers_

    def get_counterfactuals(self, x, k=5, y_desiderd=None, constrain_into_ranges=True, search_diversity=False):

        # x = x.reshape(1, -1)
        x = np.expand_dims(x, 0)
        nx = self.scaler.transform(x) if not self.pooler else self.scaler.transform(self.pooler.transform(x))

        y_val = self.b.predict(x)[0]

        cf_score = dict()
        for yc, centroids in self.n_class_centroids.items():
            if yc == y_val:
                continue
            if y_desiderd is not None and yc != y_desiderd:
                continue

            for i, c in enumerate(centroids):
                # ncfc = nx.copy()
                ncfc = nx.copy() if not self.pooler else self.scaler.transform(self.pooler.transform(x))
                ncfc[:, self.variable_features] = c
                cfc = self.scaler.inverse_transform(ncfc)
                y_cfc = self._predict(cfc)[0]
                if y_cfc != y_val:
                    if y_desiderd is not None and y_cfc != y_desiderd:
                        continue

                    if not self._respect_ranges(cfc):
                        if constrain_into_ranges:
                            cfc = self._contrain_into_ranges(cfc)
                        else:
                            continue

                    if not self._respect_categorical_features(cfc):
                        continue

                    score = self.cdist(ncfc, nx, metric=self.metric, w=self.weights).flatten()[0]
                    cf_score[len(cf_score)] = (score, cfc.flatten())

        if len(cf_score) > k and search_diversity:
            cf_list = self._get_diverse(cf_score, k)
        else:
            cf_list = self._get_closest(cf_score, k)

        if self.pooler:
            cf_list = self.pooler.inverse_transform(cf_list)

        return cf_list

    def get_prototypes(self, x, k=5, beta=0.5, constrain_into_ranges=True, search_diversity=False):

        # x = x.reshape(1, -1)
        x = np.expand_dims(x, 0)
        nx = self.scaler.transform(x) if not self.pooler else self.scaler.transform(self.pooler.transform(x))

        y_val = self.b.predict(x)[0]
        y_prob = self.b.predict_proba(x)[:, y_val][0]

        pr_score = dict()
        yc = y_val
        centroids = self.n_class_centroids[yc]
        for i, c in enumerate(centroids):

            # nprc = nx.copy()
            nprc = nx.copy() if not self.pooler else self.scaler.transform(self.pooler.transform(x))
            nprc[:, self.variable_features] = c
            prc = self.scaler.inverse_transform(nprc)

            y_prc = self._predict(prc)
            y_prc_prob = self._predict_proba(prc)[:, y_prc][0]
            if y_prc == y_val and y_prc_prob > y_prob:

                if not self._respect_ranges(prc):
                    if constrain_into_ranges:
                        prc = self._contrain_into_ranges(prc)
                    else:
                        continue

                if not self._respect_categorical_features(prc):
                    continue

                score = self._calculate_prototype_score(nx, nprc, beta=beta)
                pr_score[len(pr_score)] = (score, prc.flatten())

        if len(pr_score) > k and search_diversity:
            pr_list = self._get_diverse(pr_score, k)
        else:
            pr_list = self._get_closest(pr_score, k)

        if self.pooler:
            pr_list = self.pooler.inverse_transform(pr_list)

        return pr_list
