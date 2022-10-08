{\rtf1\ansi\ansicpg936\cocoartf1404\cocoasubrtf470
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 from scipy import sparse\
import numpy as np\
import tensorflow as tf\
import tensorflow.compat.v1 as tf\
tf.disable_v2_behavior()\
from sklearn.neighbors import NearestNeighbors, kneighbors_graph\
from scipy.linalg import solve\
from scipy.sparse import csr_matrix\
import os\
import time\
\
ROOT_PATH = '../input'\
OUTPUT_PATH = 'result'\
\
def _make_dir_if_not_exists(path):\
    if not os.path.exists (path):\
        os.mkdir (path)\
\
\
def my_barycenter_graph(X, n_neighbors, metric='cosine', reg=1e-3, n_jobs=None):\
    n_samples = X.shape[0]\
    knn = NearestNeighbors (n_neighbors = n_neighbors + 1, metric=metric, n_jobs=n_jobs).fit (X)\
    X = knn._fit_X\
    ind = knn.kneighbors (X, return_distance=False)[:, 1:]\
    #     data = barycenter_weights(X, X[ind], reg=reg)\
\
    B = np.empty ((n_samples, n_neighbors), dtype=X.dtype)\
    v = np.ones (n_neighbors, dtype=X.dtype)\
\
    # this might raise a LinalgError if G is singular and has trace\
    # zero\
    for i in range (n_samples):\
        C = X[ind[i]] - X[i]  # broadcasting\
        G = np.dot (C, C.T)\
        trace = np.trace (G)\
        if trace > 0:\
            R = reg * trace\
        else:\
            R = reg\
        G.flat[::n_neighbors + 1] += R\
        w = solve (G, v, sym_pos=True)\
        B[i, :] = w / np.sum (w)\
\
    indptr = np.arange (0, n_samples * n_neighbors + 1, n_neighbors)\
    return csr_matrix ((B.ravel (), ind.ravel (), indptr),\
                       shape=(n_samples, n_samples))\
\
def my_softmax_w(X, n_neighbors, metric='cosine',n_jobs=None):\
    n_samples = X.shape[0]\
    knn = NearestNeighbors(n_neighbors = n_neighbors + 1, metric=metric, n_jobs=n_jobs).fit(X)\
    X = knn._fit_X\
    dis, ind = knn.kneighbors(X, return_distance=True)\
    ind = ind[:, 1:]\
    data = 1-dis[:, 1:]\
    data = np.exp(data) / (np.sum(np.exp(data), axis=-1, keepdims=True) + 1e-9)\
\
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)\
    return csr_matrix((data.ravel(), ind.ravel(), indptr),\
                      shape=(n_samples, n_samples))\
\
def my_similarity_graph(X, n_neighbors, metric='cosine', reg=1e-3, n_jobs=None):\
    n_samples = X.shape[0]\
    if metric == 'cosine':\
        similarity_ = kneighbors_graph (X, n_neighbors, metric=metric, include_self=False, n_jobs=n_jobs)\
        s = similarity_.tocoo ()\
        row = s.row\
        col = s.col\
        data = np.sum (np.square (X[row] - X[col]), axis=-1)\
        similarity = csr_matrix ((data, (row, col)), shape=(n_samples, n_samples))\
    else:\
        similarity = kneighbors_graph (X, n_neighbors, metric=metric, mode='distance', include_self=False,\
                                       n_jobs=n_jobs)\
\
    return similarity\
\
\
def __init_session():\
    # gpu_options = tf.GPUOptions(\
    #     per_process_gpu_memory_fraction=GPU_MEMORY_FRAC)\
    # gpu_config = tf.ConfigProto(gpu_options=gpu_options)\
    # session = tf.Session(config=gpu_config)\
\
    config = tf.ConfigProto ()\
    config.gpu_options.allow_growth = True\
\
    session = tf.Session (config=config)\
    session.run (tf.global_variables_initializer ())\
    return session\
\
\
def _validate_batch(session, batch_manager, model):\
    valid_data = batch_manager.valid_data\
    sum_sse = 0\
    n_valid = 0\
    valid_rmse = 0\
    for m in range (0, valid_data.shape[0], 10000):\
        end_m = min (m + 10000, valid_data.shape[0])\
        u = valid_data[m:end_m, 0]\
        i = valid_data[m:end_m, 1]\
        r = valid_data[m:end_m, 2]\
        _sse = session.run (model['sse'],\
                            feed_dict=\{\
                                model['u']: u,\
                                model['i']: i,\
                                model['r']: r\
                            \})\
        sum_sse += _sse\
        n_valid += len (u)\
    valid_rmse = np.sqrt (sum_sse / n_valid)\
\
    test_data = batch_manager.test_data\
    sum_sse = 0\
    n_test = 0\
    for m in range (0, test_data.shape[0], 10000):\
        end_m = min (m + 10000, test_data.shape[0])\
        u = test_data[m:end_m, 0]\
        i = test_data[m:end_m, 1]\
        r = test_data[m:end_m, 2]\
        _sse = session.run (model['sse'],\
                            feed_dict=\{\
                                model['u']: u,\
                                model['i']: i,\
                                model['r']: r\
                            \})\
        sum_sse += _sse\
        n_test += len (u)\
    test_rmse = np.sqrt (sum_sse / n_test)\
\
    return valid_rmse, test_rmse\
\
\
def init_basic_lrmc(init, random_seed, d, lmd, lr, batch_manager):\
    n_row, n_col = batch_manager.n_user, batch_manager.n_item\
    tf.reset_default_graph ()\
    u = tf.placeholder (tf.int64, [None], name='u')\
    i = tf.placeholder (tf.int64, [None], name='i')\
    r = tf.placeholder (tf.float64, [None], name='r')\
\
    # init weights\
    tf.set_random_seed (random_seed)\
    p = tf.Variable (tf.truncated_normal ([n_row, d], 0, init, dtype=tf.float64))\
    q = tf.Variable (tf.truncated_normal ([n_col, d], 0, init, dtype=tf.float64))\
    p_lookup = tf.nn.embedding_lookup (p, u)\
    q_lookup = tf.nn.embedding_lookup (q, i)\
\
    r_hat = tf.reduce_sum (tf.multiply (p_lookup, q_lookup), 1)\
    reg_loss = 0\
    reg_loss += lmd * (tf.reduce_sum (tf.square (p_lookup)) + tf.reduce_sum (tf.square (q_lookup)))\
\
    loss = tf.reduce_sum (tf.square (r - r_hat)) + reg_loss\
    rmse = tf.sqrt (tf.reduce_mean (tf.square (r - r_hat)))\
    sse = tf.reduce_sum (tf.square (r - r_hat))\
    optimizer = tf.train.GradientDescentOptimizer (lr)\
    train_ops = optimizer.minimize (loss, var_list=[p, q])\
    return \{\
        'u': u,\
        'i': i,\
        'r': r,\
        'train_ops': train_ops,\
        'loss': loss,\
        'rmse': rmse,\
        'sse': sse,\
        'p': p,\
        'q': q,\
    \}\
\
\
def train_basic_lrmc(init, random_seed, d, lmd, lr, batch_size, batch_manager):\
    model = init_basic_lrmc (init, random_seed, d, lmd, lr, batch_manager)\
    session = __init_session ()\
    train_data = batch_manager.train_data\
\
    min_valid_rmse = 100.0\
    min_valid_iter = 0\
    min_valid_epoch = 0\
    final_test_rmse = float ('Inf')\
    iter = 0\
    last_train_rmse = 0\
    last_train_loss = float ('Inf')\
    best_p = None\
    best_q = None\
    best_u_s_value = None\
    best_i_s_value = None\
\
    for epoch in range (1, 1001):\
        sum_sse = 0\
        for m in range (0, train_data.shape[0], batch_size):\
            end_m = min (m + batch_size, train_data.shape[0])\
            u = train_data[m:end_m, 0]\
            i = train_data[m:end_m, 1]\
            r = train_data[m:end_m, 2]\
\
            p, q, loss, sse, _ = session.run ([model['p'], model['q'], model['loss'], model['sse'], model['train_ops']],\
                                              feed_dict=\{\
                                                  model['u']: u,\
                                                  model['i']: i,\
                                                  model['r']: r\
                                              \})\
            iter += 1\
            sum_sse += sse\
        train_rmse = np.sqrt (sum_sse / train_data.shape[0])\
        valid_rmse, test_rmse = _validate_batch (session, batch_manager, model)\
        if valid_rmse - min_valid_rmse < 0:\
            min_valid_rmse = valid_rmse\
            min_valid_epoch = epoch\
            final_test_rmse = test_rmse\
            best_q = q\
            best_p = p\
        elif epoch > min_valid_epoch + 10:\
            if epoch > 20:\
                print ('Early stop!')\
                break  # print('%d: %.4f %.4f %.4f/%.4f' % (epoch, train_rmse, valid_rmse, test_rmse, final_test_rmse))\
        if last_train_rmse - train_rmse < 1e-5:\
            if epoch < 20:\
                last_train_rmse = train_rmse\
            else:\
                print ('Converged!')\
                break\
        else:\
            last_train_rmse = train_rmse\
\
        if epoch % 10 == 0:\
            print ('%d: %.4f %.4f %.4f / %4f' % (epoch, train_rmse, valid_rmse, test_rmse, final_test_rmse))\
\
    return final_test_rmse, min_valid_epoch, min_valid_rmse, best_p, best_q, best_u_s_value, best_i_s_value\
\
\
def init_lrmc_manifold(method, u_row, u_col, u_value, i_row, i_col, i_value, init, random_seed, d, k, alpha, beta,\
                       lmd, lr, batch_manager):\
    n_row, n_col = batch_manager.n_user, batch_manager.n_item\
    tf.reset_default_graph ()\
    u = tf.placeholder (tf.int64, [None], name='u')\
    i = tf.placeholder (tf.int64, [None], name='i')\
    r = tf.placeholder (tf.float64, [None], name='r')\
    # init weights\
    tf.set_random_seed (random_seed)\
    p = tf.Variable (tf.truncated_normal ([n_row, d], 0, init, dtype=tf.float64))\
    q = tf.Variable (tf.truncated_normal ([n_col, d], 0, init, dtype=tf.float64))\
\
    p_lookup = tf.nn.embedding_lookup (p, u)\
    q_lookup = tf.nn.embedding_lookup (q, i)\
\
    user_manifold_reg_loss = 0\
    item_manifold_reg_loss = 0\
    if (method == 'lle'):\
        p_o = tf.gather (p, u_col[0, :])\
        w_u = tf.reshape (u_value[0, :], [-1, 1])\
        w_p = tf.reshape (w_u * p_o, [-1, k, d])\
        w_p_sum = tf.reduce_sum (w_p, axis=-2)\
\
        q_o = tf.gather (q, i_col[0, :])\
        w_i = tf.reshape (i_value[0, :], [-1, 1])\
        w_q = tf.reshape (w_i * q_o, [-1, k, d])\
        w_q_sum = tf.reduce_sum (w_q, axis=-2)\
\
        user_manifold_reg_loss = tf.reduce_sum (tf.square (p - w_p_sum))\
        item_manifold_reg_loss = tf.reduce_sum (tf.square (q - w_q_sum))\
\
    elif (method == 'lem'):\
        p_c = tf.gather (p, u_row[0, :])\
        p_o = tf.gather (p, u_col[0, :])\
        w_u = tf.reshape (u_value[0, :], [-1, 1])\
\
        q_c = tf.gather (q, i_row[0, :])\
        q_o = tf.gather (q, i_col[0, :])\
        w_i = tf.reshape (i_value[0, :], [-1, 1])\
        user_manifold_reg_loss = tf.reduce_sum (tf.reduce_sum (tf.square (p_c - p_o), -1, keepdims=True) * w_u)\
        item_manifold_reg_loss = tf.reduce_sum (tf.reduce_sum (tf.square (q_c - q_o), -1, keepdims=True) * w_i)\
\
    r_hat = tf.reduce_sum (tf.multiply (p_lookup, q_lookup), 1)\
    reg_loss = 0\
    if not alpha == 0:\
        print ('alpha')\
        reg_loss += alpha * user_manifold_reg_loss\
    if not beta == 0:\
        print ('beta')\
        reg_loss += beta * item_manifold_reg_loss\
    if not lmd == 0:\
        print ('lmd')\
        reg_loss += lmd * (tf.reduce_sum (tf.square (p_lookup)) + tf.reduce_sum (tf.square (q_lookup)))\
\
    loss = tf.reduce_sum (tf.square (r - r_hat)) + reg_loss\
    rmse = tf.sqrt (tf.reduce_mean (tf.square (r - r_hat)))\
    sse = tf.reduce_sum (tf.square (r - r_hat))\
\
    optimizer = tf.train.GradientDescentOptimizer (lr)\
    # optimizer = tf.train.MomentumOptimizer(lr, 0.9)\
    # optimizer = tf.train.AdamOptimizer(lr)\
    train_ops = optimizer.minimize (loss, var_list=[p, q])\
    return \{\
        'u': u,\
        'i': i,\
        'r': r,\
        'train_ops': train_ops,\
        'loss': loss,\
        'rmse': rmse,\
        'sse': sse,\
        'p': p,\
        'q': q,\
    \}\
\
\
def train_lrmc_manifold(method, u_row, u_col, u_value, i_row, i_col, i_value, init, random_seed, d, k, alpha, beta,\
                        lmd, lr, batch_size, batch_manager):\
    model = init_lrmc_manifold (method, u_row, u_col, u_value, i_row, i_col, i_value, init, random_seed, d, k, alpha,\
                                beta, lmd, lr, batch_manager)\
    session = __init_session ()\
    train_data = batch_manager.train_data\
\
    min_valid_rmse = 100.0\
    final_test_rmse = float ('Inf')\
    iter = 0\
    min_epoch = 0\
    min_valid_epoch = 0\
    last_train_rmse = float ('inf')\
    last_train_loss = float ('inf')\
    best_p = None\
    best_q = None\
\
    for epoch in range (1, 1001):\
        sum_sse = 0\
        for m in range (0, train_data.shape[0], batch_size):\
            end_m = min (m + batch_size, train_data.shape[0])\
            u = train_data[m:end_m, 0]\
            i = train_data[m:end_m, 1]\
            r = train_data[m:end_m, 2]\
\
            p, q, loss, sse, _ = session.run ([model['p'], model['q'], model['loss'], model['sse'], model['train_ops']],\
                                              feed_dict=\{\
                                                  model['u']: u,\
                                                  model['i']: i,\
                                                  model['r']: r\
                                              \})\
            iter += 1\
            sum_sse += sse\
        train_rmse = np.sqrt (sum_sse / train_data.shape[0])\
        valid_rmse, test_rmse = _validate_batch (session, batch_manager, model)\
        if valid_rmse - min_valid_rmse < 0:\
            min_valid_rmse = valid_rmse\
            min_valid_epoch = epoch\
            final_test_rmse = test_rmse\
            best_q = q\
            best_p = p\
        elif epoch > min_valid_epoch + 10:\
            if epoch > 30:\
                print ('Early stop!')\
                break\
        if last_train_rmse - train_rmse < 1e-5:\
            if epoch < 30:\
                last_train_rmse = train_rmse\
            else:\
                print ('Converged!')\
                break\
        else:\
            last_train_rmse = train_rmse\
\
        if epoch % 1 == 0:\
            print ('%d: %.4f %.4f %.4f / %4f' % (epoch, train_rmse, valid_rmse, test_rmse, final_test_rmse))\
\
    return final_test_rmse, min_valid_epoch, min_valid_rmse, best_p, best_q\
\
class BatchManager:\
    def __init__(self, kind, train_ratio, val_ratio, random_seed):\
        dataset_manager = DatasetManager (kind, train_ratio, val_ratio, random_seed)\
        '''\
        self.train_data = np.concatenate(\
            [\
                dataset_manager.get_train_data(),\
                dataset_manager.get_valid_data()\
            ],\
            axis=0)\
        '''\
        self.train_data = dataset_manager.get_train_data ()\
        self.valid_data = dataset_manager.get_valid_data ()\
        self.test_data = dataset_manager.get_test_data ()\
\
        self.n_user = int (\
            max (max (np.max (self.train_data[:, 0]), np.max (self.test_data[:,\
                                                              0])), np.max (self.valid_data[:, 0]))) + 1\
        self.n_item = int (\
            max (max (np.max (self.train_data[:, 1]), np.max (self.test_data[:,\
                                                              1])), np.max (self.valid_data[:, 1]))) + 1\
        self.mu = np.mean (self.train_data[:, 2])\
        self.std = np.std (self.train_data[:, 2])\
\
\
class DatasetManager:\
    KIND_MOVIELENS_100K = 'movielens-100k'\
    KIND_MOVIELENS_1M = 'movielens-1m'\
    KIND_MOVIELENS_10M = 'movielens-10m'\
\
    KIND_OBJECTS = (\
    KIND_MOVIELENS_100K, KIND_MOVIELENS_1M, KIND_MOVIELENS_10M)\
\
    def __init_data(self, detail_path, delimiter, header=False):\
        current_u = 0\
        u_dict = \{\}\
        current_i = 0\
        i_dict = \{\}\
\
        data = []\
        with open ('\{\}\{\}'.format(ROOT_PATH, detail_path), 'r') as f:\
            if header:\
                f.readline()\
\
            for line in f:\
                cols = line.strip ().split (delimiter)\
                assert len(cols) == 4\
                # cols = [float(c) for c in cols]\
                user_id = cols[0]\
                item_id = cols[1]\
                r = float (cols[2])\
                t = int (cols[3])\
\
                u = u_dict.get (user_id)\
                if u is None:\
                    u_dict[user_id] = current_u\
                    u = current_u\
                    current_u += 1\
\
                i = i_dict.get(item_id)\
                if i is None:\
                    # print(current_i)\
                    i_dict[item_id] = current_i\
                    i = current_i\
                    current_i += 1\
                data.append((u, i, r, t))\
            f.close()\
\
        data = np.array(data)\
        _make_dir_if_not_exists('data/\{\}'.format (self.kind))\
        np.save('data/\{\}/data.npy'.format (self.kind), data)\
\
    def __get_100k_data(self):\
        current_u = 0\
        u_dict = \{\}\
        current_i = 0\
        i_dict = \{\}\
\
        train_data = []\
        test_data = []\
        with open ('%s/movielens-100k-dataset/ml-100k/u1.base' % (ROOT_PATH), 'r') as f:\
            for line in f:\
                cols = line.strip ().split ('\\t')\
                assert len (cols) == 4\
                # cols = [float(c) for c in cols]\
                user_id = cols[0]\
                item_id = cols[1]\
                r = float (cols[2])\
                t = int (cols[3])\
\
                u = u_dict.get (user_id)\
                if u is None:\
                    u_dict[user_id] = current_u\
                    u = current_u\
                    current_u += 1\
\
                i = i_dict.get (item_id)\
                if i is None:\
                    # print(current_i)\
                    i_dict[item_id] = current_i\
                    i = current_i\
                    current_i += 1\
                train_data.append ((u, i, r, t))\
            f.close ()\
\
        with open ('%s/movielens-100k-dataset/ml-100k/u1.test' % (ROOT_PATH), 'r') as f:\
            for line in f:\
                cols = line.strip ().split ('\\t')\
                assert len (cols) == 4\
                # cols = [float(c) for c in cols]\
                user_id = cols[0]\
                item_id = cols[1]\
                r = float (cols[2])\
                t = int (cols[3])\
\
                u = u_dict.get (user_id)\
                if u is None:\
                    u_dict[user_id] = current_u\
                    u = current_u\
                    current_u += 1\
\
                i = i_dict.get (item_id)\
                if i is None:\
                    # print(current_i)\
                    i_dict[item_id] = current_i\
                    i = current_i\
                    current_i += 1\
                test_data.append ((u, i, r, t))\
            f.close ()\
\
        train_data_ = np.array (train_data)\
        test_data = np.array (test_data)\
        np.random.seed (self.random_seed)\
        np.random.shuffle (train_data_)\
        n_val = int (train_data_.shape[0] * (1 - self.val_ratio))\
        train_data = train_data_[:n_val]\
        valid_data = train_data_[n_val:]\
\
        _make_dir_if_not_exists ('data/\{\}'.format (self.kind))\
        np.save (self._get_npy_path ('train'), train_data)\
        np.save (self._get_npy_path ('valid'), valid_data)\
        np.save (self._get_npy_path ('test'), test_data)\
\
    def _init_data(self):\
        if self.kind == self.KIND_MOVIELENS_100K:\
            self.__init_data('/movielens-100k-dataset/ml-100k/u.data', '\\t')\
        elif self.kind == self.KIND_MOVIELENS_1M:\
            self.__init_data('/movielens-1m-dataset/ratings.dat', '::')\
        elif self.kind == self.KIND_MOVIELENS_10M:\
            self.__init_data('/ml-10m/ratings.dat', '::')\
        else:\
            raise NotImplementedError ()\
\
    def _split_data(self):\
        data = self.data\
        np.random.seed(self.random_seed)\
        np.random.shuffle(data)\
\
        n_train = int(data.shape[0] * self.train_ratio)\
        n_valid = int(n_train * (1 - self.val_ratio))\
        train_data = data[:n_valid]\
        valid_data = data[n_valid:n_train]\
        test_data = data[n_train:]\
\
        np.save(self._get_npy_path ('train'), train_data)\
        np.save(self._get_npy_path ('valid'), valid_data)\
        np.save(self._get_npy_path ('test'), test_data)\
\
    def _load_base_data(self):\
        return np.load('data/%s/data.npy'%(self.kind))\
\
    def _get_npy_path(self, split_kind):\
        return 'data/%s/%s%s.npy' % (self.kind, self.random_seed,\
                                  split_kind)\
\
    def __init__(self, kind, train_ratio=0.9, val_ratio=0.1, random_seed=0):\
        if kind not in self.KIND_OBJECTS:\
            raise NotImplementedError ()\
        else:\
            self.kind = kind\
        _make_dir_if_not_exists('data')\
        self.train_ratio = train_ratio\
        self.val_ratio = val_ratio\
        self.random_seed = random_seed\
        if kind == self.KIND_MOVIELENS_100K:\
            self.__get_100k_data ()\
        else:\
            if not os.path.exists('data/\{\}/data.npy'.format(self.kind)):\
                self._init_data()\
            self.data = self._load_base_data()\
\
        if not os.path.exists(self._get_npy_path ('train')) or not os.path.exists (\
            self._get_npy_path('valid')) or not os.path.exists (\
            self._get_npy_path('test')):\
            self._split_data()\
\
        self.train_data = np.load(self._get_npy_path ('train'))\
        self.valid_data = np.load(self._get_npy_path ('valid'))\
        self.test_data = np.load(self._get_npy_path ('test'))\
\
    def get_train_data(self):\
        return self.train_data\
\
    def get_valid_data(self):\
        return self.valid_data\
\
    def get_test_data(self):\
        return self.test_data\
\
\
\
if __name__ == '__main__':\
    kind = DatasetManager.KIND_MOVIELENS_100K\
    train_ratio = 0.8\
#     kind = DatasetManager.KIND_MOVIELENS_1M\
#     train_ratio = 0.9\
    # kind = DatasetManager.KIND_MOVIELENS_10M\
\
    val_ratio = 0.05\
    init = 0.01\
    \
    d_list = np.array([512])\
    k_list = np.array([40])\
    method = 'softmax'\
    \
    _make_dir_if_not_exists('%s'%(OUTPUT_PATH))\
    result_fd = open('%s/%s_test_result_split%sv%s_init%s_d%s_k%s_%s.csv' % (\
    OUTPUT_PATH, kind, train_ratio, val_ratio, init, d_list, k_list, method), 'w')\
    result_fd.write('random, d, item_k, user_k, alpha, beta, l, lr, batch_size, test_rmse, epoch, val_rmse\\n')\
    result_fd.flush()\
    result = []\
    for random_seed in [2]:\
        batchManager = BatchManager(kind, train_ratio, val_ratio, random_seed)\
        train_data = batchManager.train_data\
        print(len(train_data))\
        print(train_data[0])\
        print(batchManager.n_user)\
        print(batchManager.n_item)\
        a_train = np.zeros([batchManager.n_user, batchManager.n_item])\
        for u, i, r, ti in train_data:\
            a_train[int(u), int(i)] = float(r)\
        a_train_item = np.transpose(a_train)\
        print('%s: %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), a_train.shape))\
        \
        for k in k_list:\
            user_k = k\
            item_k = k\
            \
            if method == 'basic':\
                print('skip!')\
            elif (method == 'lle'):\
                print('no job before: %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))\
                user_similarity = my_barycenter_graph (a_train, user_k,n_jobs=1)\
                print('LLE user complete: %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))\
                item_similarity = my_barycenter_graph (a_train_item, item_k,n_jobs=1)\
                print('LLE item complete: %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))\
            elif (method == 'softmax'):\
                print('no job before: %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))\
                user_similarity = my_softmax_w(a_train, user_k,n_jobs=1)\
                print('Softmax user complete: %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))\
                item_similarity = my_softmax_w(a_train_item, item_k,n_jobs=1)\
                print('Softmax item complete: %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))\
            else:\
                NotImplementedError()\
            for d in d_list:\
                for alpha, beta, lmd in [[1,5,0.02]]:\
                    for lr, batch_size in [[0.001, 10000]]:\
                        print('%s: Start!' % (time.strftime ("%Y-%m-%d %H:%M:%S", time.localtime ())))\
                        print('param: %d %d %d %d %s %s %s %s %d' % (\
                        random_seed, d, user_k, item_k, alpha, beta, lmd, lr, batch_size))\
                        if (method == 'lle')|(method == 'softmax'):\
                            u_row = np.reshape(user_similarity.tocoo ().row, [1, -1])\
                            u_col = np.reshape(user_similarity.tocoo ().col, [1, -1])\
                            u_value = np.reshape(user_similarity.tocoo ().data, [1, -1])\
                            i_row = np.reshape(item_similarity.tocoo ().row, [1, -1])\
                            i_col = np.reshape(item_similarity.tocoo ().col, [1, -1])\
                            i_value = np.reshape(item_similarity.tocoo ().data, [1, -1])\
                            print(u_value)\
                            print(i_value)\
                            \
                            final_test_rmse, min_epoch, min_val_rmse, p, q = train_lrmc_manifold('lle', u_row, u_col, u_value, i_row, i_col, i_value, init,\
                                                                                                        random_seed, d, k,\
                                                                                                        alpha, beta, lmd, lr,\
                                                                                                        batch_size, batchManager)\
                        elif method == 'basic':\
                            final_test_rmse, min_epoch, min_val_rmse, p, q, w_u, w_i = train_basic_lrmc(init, random_seed,d, lmd,lr,batch_size,batchManager)\
                        print ('%s param: %d %d %d %d %s %s %s %s %d result:%.6f/%d/%.6f' % (\
                            time.strftime ("%Y-%m-%d %H:%M:%S", time.localtime ()),\
                            random_seed, d, user_k, item_k, alpha, beta, lmd, lr, batch_size,\
                            final_test_rmse,\
                            min_epoch, min_val_rmse))\
                        result_fd.write ('%d, %d, %d, %d, %s, %s, %s, %s, %d, %.6f, %d, %.6f\\n' % (\
                                random_seed, d, user_k, item_k, alpha, beta, lmd, lr, batch_size,\
                                final_test_rmse,\
                                min_epoch, min_val_rmse))\
                        result_fd.flush ()\
                        result.append (final_test_rmse)\
\
    result_fd.write ('%.5f\\n' % np.mean (result))\
    result.clear ()}