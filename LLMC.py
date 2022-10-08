##### from scipy import sparse
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
#
# tf.disable_v2_behavior()
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.linalg import solve
from scipy.sparse import csr_matrix, coo_matrix
from scipy import sparse
from sklearn.manifold import SpectralEmbedding
import os
import time
import h5py

ROOT_PATH = 'input'
OUTPUT_PATH = 'result'


def _make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()
    return out


def get_similarity_matrix(X, method='reconstruct', K=10, metric='cosine', reg=1e-3, n_jobs=1):
    n_samples = X.shape[0]
    if method == 'average':
        s = kneighbors_graph(X, n_neighbors=K, metric=metric, include_self=False, n_jobs=n_jobs)
        W = s.multiply(1 / K).tocoo()
    elif method == 'softmax':
        s = kneighbors_graph(X, n_neighbors=K, mode='distance', metric=metric, include_self=False, n_jobs=n_jobs)
        rows = s.tocoo().row
        cols = s.tocoo().col
        data = 1 - s.tocoo().data
        exp_data = np.reshape(np.exp(data), [n_samples, K])
        soft_data = exp_data / (np.sum(exp_data, axis=-1, keepdims=True))
        W = coo_matrix((soft_data.ravel(), (rows, cols)), shape=(n_samples, n_samples))
    elif method == 'reconstruct':
        s = kneighbors_graph(X, n_neighbors=K, metric=metric, include_self=False, n_jobs=n_jobs)
        B = np.empty((n_samples, K), dtype=X.dtype)
        v = np.ones(K, dtype=X.dtype)
        rows = s.tocoo().row
        cols = s.tocoo().col
        for i in range(n_samples):
            C = X[cols[i * K:i * K + K]] - X[rows[i * K:i * K + K]]
            G = np.array(C.dot(C.transpose()).todense(), dtype=np.float64)
            trace = np.trace(G)
            if trace > 0:
                R = reg * trace
            else:
                R = reg
            G.flat[::K + 1] += R
            w = solve(G, v, sym_pos=True)
            B[i, :] = w / np.sum(w)
        W = coo_matrix((B.ravel(), (rows, cols)), shape=(n_samples, n_samples))
    else:
        raise NotImplementedError()
    return W


def __init_session():
    # gpu_options = tf.GPUOptions(
    #     per_process_gpu_memory_fraction=GPU_MEMORY_FRAC)
    # gpu_config = tf.ConfigProto(gpu_options=gpu_options)
    # session = tf.Session(config=gpu_config)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())
    return session


def _validate_batch(session, batch_manager, model):
    valid_data = batch_manager.valid_data
    sum_sse = 0
    n_valid = 0
    valid_rmse = 0
    if not valid_data.size == 0:
        for m in range(0, valid_data.shape[0], 100000):
            end_m = min(m + 100000, valid_data.shape[0])
            u = valid_data[m:end_m, 0]
            i = valid_data[m:end_m, 1]
            r = valid_data[m:end_m, 2]
            _sse = session.run(model['sse'],
                               feed_dict={
                                   model['u']: u,
                                   model['i']: i,
                                   model['r']: r
                               })
            sum_sse += _sse
            n_valid += len(u)
        valid_rmse = np.sqrt(sum_sse / n_valid)

    test_data = batch_manager.test_data
    sum_sse = 0
    n_test = 0
    for m in range(0, test_data.shape[0], 100000):
        end_m = min(m + 100000, test_data.shape[0])
        u = test_data[m:end_m, 0]
        i = test_data[m:end_m, 1]
        r = test_data[m:end_m, 2]
        _sse = session.run(model['sse'],
                           feed_dict={
                               model['u']: u,
                               model['i']: i,
                               model['r']: r
                           })
        sum_sse += _sse
        n_test += len(u)
    test_rmse = np.sqrt(sum_sse / n_test)

    return valid_rmse, test_rmse


def init_llmc(train_method, s_u, s_i, init, random_seed, d, k, alpha, beta,
              lmd_u, lmd_v, base_lr, batch_manager):
    n_row, n_col = batch_manager.n_user, batch_manager.n_item
    tf.reset_default_graph()
    u = tf.placeholder(tf.int64, [None], name='u')
    i = tf.placeholder(tf.int64, [None], name='i')
    r = tf.placeholder(tf.float64, [None], name='r')
    lr = tf.Variable(base_lr, name='lr', trainable=False)
    # init weights
    tf.set_random_seed(random_seed)
    p = tf.Variable(tf.truncated_normal([n_row, d], 0, init, dtype=tf.float64))
    q = tf.Variable(tf.truncated_normal([n_col, d], 0, init, dtype=tf.float64))

    p_lookup = tf.nn.embedding_lookup(p, u)
    q_lookup = tf.nn.embedding_lookup(q, i)

    user_manifold_reg_loss = 0
    item_manifold_reg_loss = 0

    indices = np.array([s_u.row, s_u.col]).transpose()
    w_u = tf.SparseTensor(indices, s_u.data, s_u.shape)
    p_hat = tf.sparse_tensor_dense_matmul(w_u, p)
    user_manifold_reg_loss = tf.reduce_sum(tf.square(p - p_hat))

    indices = np.array([s_i.row, s_i.col]).transpose()
    w_i = tf.SparseTensor(indices, s_i.data, s_i.shape)
    q_hat = tf.sparse_tensor_dense_matmul(w_i, q)
    item_manifold_reg_loss = tf.reduce_sum(tf.square(q - q_hat))

    r_hat = tf.reduce_sum(tf.multiply(p_lookup, q_lookup), 1)
    reg_loss = 0
    if not alpha == 0:
        print('alpha')
        reg_loss += alpha * user_manifold_reg_loss
    if not beta == 0:
        print('beta')
        reg_loss += beta * item_manifold_reg_loss
    if (not lmd_u == 0) | (not lmd_v == 0):
        print('lmd')
    reg_loss += lmd_u * tf.reduce_sum(tf.square(p_lookup)) + lmd_v * tf.reduce_sum(tf.square(q_lookup))

    loss = 0.5 * tf.reduce_sum(tf.square(r - r_hat)) + 0.5 * reg_loss
    rmse = tf.sqrt(tf.reduce_mean(tf.square(r - r_hat)))
    sse = tf.reduce_sum(tf.square(r - r_hat))
    if train_method == 'adam':
        optimizer = tf.train.AdamOptimizer(lr)
    elif train_method == 'gd':
        optimizer = tf.train.GradientDescentOptimizer(lr)
    train_ops = optimizer.minimize(loss, var_list=[p, q])
    return {
        'u': u,
        'i': i,
        'r': r,
        'train_ops': train_ops,
        'loss': loss,
        'rmse': rmse,
        'sse': sse,
        'p': p,
        'q': q,
        'lr': lr
    }


def train_llmc(epoches, train_method, s_u, s_i, init, random_seed, d, k, alpha, beta,
               lmd_u, lmd_v, lr, batch_size, batch_manager):
    model = init_llmc(train_method, s_u, s_i, init, random_seed, d, k, alpha,
                      beta, lmd_u, lmd_v, lr, batch_manager)
    session = __init_session()
    train_data = batch_manager.train_data

    min_valid_rmse = 100.0
    final_test_rmse = float('Inf')
    min_epoch = 0
    min_valid_epoch = 0
    last_train_rmse = float('inf')
    last_train_loss = float('inf')
    best_p = None
    best_q = None

    for epoch in range(1, epoches + 1):
        sum_sse = 0
        loss_sum = 0
        #         np.random.shuffle(train_data)
        for m in range(0, train_data.shape[0], batch_size):
            end_m = min(m + batch_size, train_data.shape[0])
            u = train_data[m:end_m, 0]
            i = train_data[m:end_m, 1]
            r = train_data[m:end_m, 2]

            p, q, loss, sse, _ = session.run([model['p'], model['q'], model['loss'], model['sse'], model['train_ops']],
                                             feed_dict={
                                                 model['u']: u,
                                                 model['i']: i,
                                                 model['r']: r,
                                                 model['lr']: lr
                                             })
            sum_sse += sse
            loss_sum += loss
        #         if last_train_loss < loss_sum:
        #             lr = lr * 0.9
        #             print('%d epoch change! %.6f'%(epoch, lr))
        last_train_loss = loss_sum
        train_rmse = np.sqrt(sum_sse / train_data.shape[0])
        valid_rmse, test_rmse = _validate_batch(session, batch_manager, model)
        if not valid_rmse == 0:
            if valid_rmse - min_valid_rmse < 0:
                min_valid_rmse = valid_rmse
                min_valid_epoch = epoch
                final_test_rmse = test_rmse
                best_q = q
                best_p = p
            elif epoch > min_valid_epoch + 20:
                print('early stop')
                break
        else:
            final_test_rmse = test_rmse

        if epoch % 10 == 0:
            print(
                '%d: %.4f %.6f %.6f %.6f / %6f' % (epoch, loss_sum, train_rmse, valid_rmse, test_rmse, final_test_rmse))

    return final_test_rmse, min_valid_epoch, min_valid_rmse, best_p, best_q


class BatchManager:
    def __init__(self, kind, train_ratio, val_ratio, random_seed):
        dataset_manager = DatasetManager(kind, train_ratio, val_ratio, random_seed)
        '''
        self.train_data = np.concatenate(
            [
                dataset_manager.get_train_data(),
                dataset_manager.get_valid_data()
            ],
            axis=0)
        '''
        self.train_data = dataset_manager.get_train_data()
        self.valid_data = dataset_manager.get_valid_data()
        self.test_data = dataset_manager.get_test_data()

        if self.valid_data.size == 0:
            self.n_user = int(max(np.max(self.train_data[:, 0]), np.max(self.test_data[:, 0]))) + 1
            self.n_item = int(max(np.max(self.train_data[:, 1]), np.max(self.test_data[:, 1]))) + 1
        else:
            self.n_user = int(
                max(max(np.max(self.train_data[:, 0]), np.max(self.test_data[:,
                                                              0])), np.max(self.valid_data[:, 0]))) + 1
            self.n_item = int(
                max(max(np.max(self.train_data[:, 1]), np.max(self.test_data[:,
                                                              1])), np.max(self.valid_data[:, 1]))) + 1
        self.mu = np.mean(self.train_data[:, 2])
        self.std = np.std(self.train_data[:, 2])


class DatasetManager:
    KIND_MOVIELENS_100K = 'ml-100k'
    KIND_MOVIELENS_1M = 'ml-1m'
    KIND_MOVIELENS_10M = 'ml-10m'
    KIND_NETFLIX = 'netflix'
    KIND_FLIXSTER = 'flixster'
    KIND_DOUBAN = 'douban'
    KIND_YAHOO = 'yahoo'
    KIND_OBJECTS = (
        KIND_YAHOO, KIND_FLIXSTER, KIND_DOUBAN, KIND_MOVIELENS_100K, KIND_MOVIELENS_1M, KIND_MOVIELENS_10M,
        KIND_NETFLIX)

    def __init_3000_data(self, path):
        M = load_matlab_file('{}{}'.format(ROOT_PATH, path), 'M')
        print(np.shape(M))
        Otraining = load_matlab_file('{}{}'.format(ROOT_PATH, path), 'Otraining')
        Otest = load_matlab_file('{}{}'.format(ROOT_PATH, path), 'Otest')

        train_uir = np.append(np.argwhere(Otraining != 0), np.reshape(M[Otraining != 0], [-1, 1]), axis=-1)
        train_data = np.append(train_uir, np.zeros([train_uir.shape[0], 1]), axis=-1)

        test_uir = np.append(np.argwhere(Otest != 0), np.reshape(M[Otest != 0], [-1, 1]), axis=-1)
        test_data = np.append(test_uir, np.zeros([test_uir.shape[0], 1]), axis=-1)
        return np.array(train_data), np.array(test_data)

    def _get_fdy_split_data(self, path):
        np.random.seed(self.random_seed)
        train_data_, test_data = self.__init_3000_data(path)
        np.random.shuffle(train_data_)
        n_val = int(train_data_.shape[0] * (1 - self.val_ratio))
        train_data = train_data_[:n_val]
        valid_data = train_data_[n_val:]
        # test_data = self.__init_data('/ml-100k/u1.test', '\t')

        _make_dir_if_not_exists('data/{}'.format(self.kind))
        np.save(self._get_npy_path('train'), train_data)
        np.save(self._get_npy_path('valid'), valid_data)
        np.save(self._get_npy_path('test'), test_data)

    def __init_data(self, detail_path, delimiter, header=False):
        current_u = 0
        u_dict = {}
        current_i = 0
        i_dict = {}

        data = []
        with open('{}{}'.format(ROOT_PATH, detail_path), 'r') as f:
            if header:
                f.readline()

            for line in f:
                cols = line.strip().split(delimiter)
                assert len(cols) == 4
                # cols = [float(c) for c in cols]
                user_id = cols[0]
                item_id = cols[1]
                r = float(cols[2])
                t = int(cols[3])

                u = u_dict.get(user_id)
                if u is None:
                    u_dict[user_id] = current_u
                    u = current_u
                    current_u += 1

                i = i_dict.get(item_id)
                if i is None:
                    # print(current_i)
                    i_dict[item_id] = current_i
                    i = current_i
                    current_i += 1
                data.append((u, i, r, t))
            f.close()

        data = np.array(data)
        _make_dir_if_not_exists('data/{}'.format(self.kind))
        np.save('data/{}/data.npy'.format(self.kind), data)

    def __get_100k_data(self):
        current_u = 0
        u_dict = {}
        current_i = 0
        i_dict = {}

        train_data = []
        test_data = []
        with open('%s/ml-100k/u1.base' % (ROOT_PATH), 'r') as f:
            for line in f:
                cols = line.strip().split('\t')
                assert len(cols) == 4
                # cols = [float(c) for c in cols]
                user_id = cols[0]
                item_id = cols[1]
                r = float(cols[2])
                t = int(cols[3])

                u = u_dict.get(user_id)
                if u is None:
                    u_dict[user_id] = current_u
                    u = current_u
                    current_u += 1

                i = i_dict.get(item_id)
                if i is None:
                    # print(current_i)
                    i_dict[item_id] = current_i
                    i = current_i
                    current_i += 1
                train_data.append((u, i, r, t))
            f.close()

        with open('%s/ml-100k/u1.test' % (ROOT_PATH), 'r') as f:
            for line in f:
                cols = line.strip().split('\t')
                assert len(cols) == 4
                # cols = [float(c) for c in cols]
                user_id = cols[0]
                item_id = cols[1]
                r = float(cols[2])
                t = int(cols[3])

                u = u_dict.get(user_id)
                if u is None:
                    u_dict[user_id] = current_u
                    u = current_u
                    current_u += 1

                i = i_dict.get(item_id)
                if i is None:
                    # print(current_i)
                    i_dict[item_id] = current_i
                    i = current_i
                    current_i += 1
                test_data.append((u, i, r, t))
            f.close()

        train_data_ = np.array(train_data)
        test_data = np.array(test_data)
        np.random.seed(self.random_seed)
        np.random.shuffle(train_data_)
        n_val = int(train_data_.shape[0] * (1 - self.val_ratio))
        train_data = train_data_[:n_val]
        valid_data = train_data_[n_val:]

        _make_dir_if_not_exists('data/{}'.format(self.kind))
        np.save(self._get_npy_path('train'), train_data)
        np.save(self._get_npy_path('valid'), valid_data)
        np.save(self._get_npy_path('test'), test_data)

    def _init_data(self):
        if self.kind == self.KIND_MOVIELENS_1M:
            self.__init_data('/ml-1m/ratings.dat', '::')
        elif self.kind == self.KIND_MOVIELENS_10M:
            self.__init_data('/ml-10m/ratings.dat', '::')
        else:
            raise NotImplementedError()

    def _split_data(self):
        data = self.data
        np.random.seed(self.random_seed)
        np.random.shuffle(data)

        n_train = int(data.shape[0] * self.train_ratio)
        n_valid = int(n_train * (1 - self.val_ratio))
        train_data = data[:n_valid]
        valid_data = data[n_valid:n_train]
        test_data = data[n_train:]
        _make_dir_if_not_exists('data/{}'.format(self.kind))
        np.save(self._get_npy_path('train'), train_data)
        np.save(self._get_npy_path('valid'), valid_data)
        np.save(self._get_npy_path('test'), test_data)

    def _load_base_data(self):
        return np.load('data/%s/data.npy' % (self.kind))

    def _get_npy_path(self, split_kind):
        return 'data/%s/%s%s.npy' % (self.kind, self.random_seed,
                                     split_kind)

    def __init__(self, kind, train_ratio=0.9, val_ratio=0.1, random_seed=0):
        if kind not in self.KIND_OBJECTS:
            raise NotImplementedError()
        else:
            self.kind = kind
        _make_dir_if_not_exists('data')
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        if kind == self.KIND_MOVIELENS_100K:
            self.__get_100k_data()
        elif kind == self.KIND_DOUBAN:
            self._get_fdy_split_data('/douban/training_test_dataset.mat')
        elif kind == self.KIND_FLIXSTER:
            self._get_fdy_split_data('/flixster/training_test_dataset_10_NNs.mat')
        elif kind == self.KIND_YAHOO:
            self._get_fdy_split_data('/yahoomusic/training_test_dataset_10_NNs.mat')
        elif kind == self.KIND_NETFLIX:
            self.data = np.load('%s/netflix/data.npy' % ROOT_PATH)
            _make_dir_if_not_exists('data/{}'.format(self.kind))
        else:
            if not os.path.exists('data/{}/data.npy'.format(self.kind)):
                self._init_data()
            self.data = self._load_base_data()

        if not os.path.exists(self._get_npy_path('train')) or not os.path.exists(
                self._get_npy_path('valid')) or not os.path.exists(
            self._get_npy_path('test')):
            self._split_data()

        self.train_data = np.load(self._get_npy_path('train'))
        self.valid_data = np.load(self._get_npy_path('valid'))
        self.test_data = np.load(self._get_npy_path('test'))

    def get_train_data(self):
        return self.train_data

    def get_valid_data(self):
        return self.valid_data

    def get_test_data(self):
        return self.test_data


if __name__ == '__main__':
    #     kind = DatasetManager.KIND_YAHOO
    #     train_ratio = 0.9
    #     batch_size = 500
    #     kind = DatasetManager.KIND_FLIXSTER
    #     train_ratio = 0.9
    #     batch_size = 2000
    #     kind = DatasetManager.KIND_DOUBAN
    #     train_ratio = 0.9
    #     batch_size = 10000
    kind = DatasetManager.KIND_MOVIELENS_100K
    train_ratio = 0.8
    batch_size = 10000
    #     kind = DatasetManager.KIND_MOVIELENS_1M
    #     train_ratio = 0.9
    #     batch_size = 20000
    #     kind = DatasetManager.KIND_MOVIELENS_10M
    #     train_ratio = 0.9
    #     batch_size = 20000
    #     kind = DatasetManager.KIND_NETFLIX
    #     train_ratio = 0.9
    #     batch_size = 100000

    val_ratio = 0.05
    epoches = 5000
    init = 0.01

    d_list = np.array([32])
    k_list = np.array([40])
    lr = 0.001

    coe_method = 'reconstruct'
    train_method = 'gd'

    _make_dir_if_not_exists('%s' % (OUTPUT_PATH))
    result_fd = open('%s/%s_test_result_split%sv%s_init%s_epoches%s_d%s_k%s_%s_%s.csv' % (
        OUTPUT_PATH, kind, train_ratio, val_ratio, init, epoches, d_list, k_list, coe_method, train_method), 'w')
    result_fd.write(
        'random, d, item_k, user_k, alpha, beta, lmd_u, lmd_v, lr, batch_size, test_rmse, epoch, val_rmse\n')
    result_fd.flush()
    result = []
    for random_seed in range(5):
        batchManager = BatchManager(kind, train_ratio, val_ratio, random_seed)
        for k in k_list:
            user_k = k
            item_k = k
            r_i = csr_matrix(((batchManager.train_data[:, 2]).astype(np.float64).tolist(), (
                batchManager.train_data[:, 1].astype(np.int32).tolist(),
                batchManager.train_data[:, 0].astype(np.int32).tolist())),
                             [batchManager.n_item, batchManager.n_user])
            r_u = csr_matrix(((batchManager.train_data[:, 2]).astype(np.float64).tolist(), (
                batchManager.train_data[:, 0].astype(np.int32).tolist(),
                batchManager.train_data[:, 1].astype(np.int32).tolist())),
                             [batchManager.n_user, batchManager.n_item])
            s_u = get_similarity_matrix(r_u, method=coe_method, K=user_k)
            s_i = get_similarity_matrix(r_i, method=coe_method, K=item_k)
            for d in d_list:
                for alpha in [1]:
                    for beta in [5]:
                        for lmd_u in [0.01]:
                            lmd_v = lmd_u
                            print('%s: Start!' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                            print('param: %d %d %d %d %s %s %s %s %s %d' % (
                                random_seed, d, user_k, item_k, alpha, beta, lmd_u, lmd_v, lr, batch_size))

                            final_test_rmse, min_epoch, min_val_rmse, p, q = train_llmc(epoches, train_method, s_u, s_i,
                                                                                        init, random_seed, d, k,
                                                                                        alpha, beta, lmd_u, lmd_v, lr,
                                                                                        batch_size, batchManager)
                            print('%s param: %d %d %d %d %s %s %s %s %s %d result:%.6f/%d/%.6f' % (
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                random_seed, d, user_k, item_k, alpha, beta, lmd_u, lmd_v, lr, batch_size,
                                final_test_rmse,
                                min_epoch, min_val_rmse))
                            result_fd.write('%d, %d, %d, %d, %s, %s, %s, %s,%s, %d, %.6f, %d, %.6f\n' % (
                                random_seed, d, user_k, item_k, alpha, beta, lmd_u, lmd_v, lr, batch_size,
                                final_test_rmse,
                                min_epoch, min_val_rmse))
                            result_fd.flush()
                            result.append(final_test_rmse)
