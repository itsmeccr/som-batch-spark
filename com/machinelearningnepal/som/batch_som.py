from pyspark import SparkContext, Row, SQLContext
from pyspark.ml.linalg import Vectors

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches

__author__ = 'topsykretts'


def find_bmu(row_t, net):
        """
            Find the best matching unit for a given vector, row_t, in the SOM
            Returns: a (bmu, bmu_idx) tuple where bmu is the high-dimensional Best Matching Unit
                     and bmu_idx is the index of this vector in the SOM
        """
        net_size = np.shape(net)
        x_size = net_size[0]
        y_size = net_size[1]
        num_features = net_size[2]
        bmu_idx = np.array([0, 0])
        # set the initial minimum distance to a huge number
        min_dist = np.iinfo(np.int).max
        # calculate the high-dimensional distance between each neuron and the input
        # for (k = 1,..., K)
        for x in range(x_size):
            for y in range(y_size):
                weight_k = net[x, y, :].reshape(1, num_features)
                # compute distances dk using Eq. (2)
                sq_dist = np.sum((weight_k - row_t) ** 2)
                # compute winning node c using Eq. (3)
                if sq_dist < min_dist:
                    min_dist = sq_dist
                    bmu_idx = np.array([x, y])
        # get vector corresponding to bmu_idx
        bmu = net[bmu_idx[0], bmu_idx[1], :].reshape(1, num_features)
        return bmu, bmu_idx


def calculate_influence(distance, radius):
        return np.exp(-distance / (radius ** 2))


class SOM:
    """
    Implementation of Batch SOM
    """

    def __init__(self, net_x_dim, net_y_dim, num_features):
        self.network_dimensions = np.array([net_x_dim, net_y_dim])
        self.init_radius = min(self.network_dimensions[0], self.network_dimensions[1])
        # initialize weight vectors
        self.num_features = num_features
        self.initialize()

    def initialize(self):
        self.net = np.random.random((self.network_dimensions[0], self.network_dimensions[1], self.num_features))

    def train(self, df, num_epochs, resetWeights=False):
        """

        :param df: input dataframe for training with "features" column
        :param num_epochs: for how many epochs should the dataframe be trained
        :param resetWeights: should the weights be randomized for next training
        :return:
        """
        if resetWeights:
            self.initialize()
        self.time_constant = num_epochs / np.log(self.init_radius)
        # visualization
        if self.num_features == 3:
            fig = plt.figure()
        else:
            fig = None
        rdd = df.rdd.cache()
        sc = SparkContext.getOrCreate()
        # for (epoch = 1,..., Nepochs)
        for i in range(1, num_epochs + 1):
            radius = self.decay_radius(i)
            vis_interval = int(num_epochs/10)
            if i % vis_interval == 0:
                if fig is not None:
                    self.show_plot(fig, i/vis_interval, i)
                print("SOM training epoches %d" % i)
                print("neighborhood radius ", radius)
                # print(self.net)
                print("-------------------------------------")
            broadcast_net = sc.broadcast(self.net)

            def train_partition_wrapper(x_size, y_size, num_features):

                def train_partition(partition_rows):
                    partition_net = broadcast_net.value
                    part_sum_numerator = np.array(np.zeros([x_size, y_size, num_features]))
                    part_sum_denominator = np.array(np.zeros([x_size, y_size, 1]))
                    for row_t in partition_rows:
                        bmu, bmu_idx = find_bmu(row_t['features'], partition_net)
                        for x in range(x_size):
                            for y in range(y_size):
                                w_dist = np.sum((np.array([x, y]) - bmu_idx) ** 2)
                                # if the distance is within the current neighbourhood radius
                                if w_dist <= radius ** 2:
                                    # update weight vectors wk using Eq. (3)
                                    influence = calculate_influence(w_dist, radius)
                                    part_sum_denominator[x, y, :] = part_sum_denominator[x, y, :] + influence
                                    new_w = influence * row_t['features']
                                    part_sum_numerator[x, y, :] = part_sum_numerator[x, y, :] + new_w
                    yield Row(num=part_sum_numerator, den=part_sum_denominator)

                return train_partition

            epoch_sum_num = np.array(np.zeros([self.network_dimensions[0], self.network_dimensions[1], self.num_features]))
            epoch_sum_den = np.array(np.zeros([self.network_dimensions[0], self.network_dimensions[1], self.num_features]))
            part_sum_rdd = rdd.mapPartitions(train_partition_wrapper(self.network_dimensions[0], self.network_dimensions[1], self.num_features))
            for row in part_sum_rdd.collect():
                epoch_sum_num += row['num']
                epoch_sum_den += row['den']
            self.net = epoch_sum_num/epoch_sum_den

        if fig is not None:
            plt.show()

    def predict(self, df):
        # find its Best Matching Unit
        column_names = df.columns

        def prediction_wrapper(net):
            def prediction_map_func(row):
                cols_map = {}
                for col in column_names:
                    cols_map[col] = row[col]
                bmu, bmu_idx = find_bmu(row['features'], net)
                cols_map["bmu"] = Vectors.dense(bmu[0])
                cols_map["bmu_idx"] = Vectors.dense(bmu_idx)
                return Row(**cols_map)
            rdd_prediction = df.rdd.map(lambda row: prediction_map_func(row))
            # getting existing sparkContext
            sc = SparkContext.getOrCreate()
            sqlContext = SQLContext(sc)
            return sqlContext.createDataFrame(rdd_prediction)
        return prediction_wrapper(self.net)



    def decay_radius(self, iteration):
        return self.init_radius * np.exp(-iteration / self.time_constant)

    def show_plot(self, fig, position, epoch):
        # setup axes
        ax = fig.add_subplot(2, 5, position, aspect="equal")
        ax.set_xlim((0, self.net.shape[0] + 1))
        ax.set_ylim((0, self.net.shape[1] + 1))
        ax.set_title('Ep: %d' % epoch)

        # plot the rectangles
        for x in range(1, self.net.shape[0] + 1):
            for y in range(1, self.net.shape[1] + 1):
                ax.add_patch(patches.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                               facecolor=self.net[x - 1, y - 1, :],
                                               edgecolor='none'))

