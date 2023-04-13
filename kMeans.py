import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from sklearn import datasets


def generate_random_points(point_count, min_range, max_range, centers):
    return datasets.make_blobs(n_samples=point_count, centers=centers, n_features=2,
                               cluster_std=(math.floor((np.random.random() * 31) + 10)),
                               center_box=(min_range, max_range), shuffle=True, random_state=None)


def main():
    print("K-Means Clustering \nEnter the total number of points you want to be generated")
    point_count = int(input("Number of points: "))
    print("Enter the range from which you want the points to be generated between")
    min_range = int(input("Min: "))
    max_range = int(input("Max: "))
    print("Enter the number of cluster you want")
    cluster_count = int(input("Number of clusters: "))
    print("Enter the threshold you want for identifying if a point belongs to a cluster")
    cluster_threshold = int(input("Cluster threshold: "))
    print("Enter the number of times you would like the algorithm to loop")
    loop_count = int(input("Loop count: "))

    random_dataset = generate_random_points(point_count, min_range, max_range, cluster_count)
    point_list = []

    for i in range(point_count):
        point_list.append(Point(random_dataset[0][i][0], random_dataset[0][i][1], i))

    x = KMeansClustering(point_list, cluster_count, cluster_threshold)
    x.start(loop_count)


class KMeansClustering():

    def __init__(self, point_set, cluster_count, max_threshold):
        self.cluster_count = cluster_count
        self.max_threshold = max_threshold
        self.point_set = point_set
        self.clusters = []
        self.last_cluster = []
        self.generate_seed_points(point_set, cluster_count, max_threshold)
        self.assign_points(point_set, self.clusters)
        self.stabalized = False
        self.convergence_threshold = 1e-6  # Convergence threshold value for checking center movement
        print("End of init: " + str(self.clusters))
        print(round(max_threshold * 0.33, 0))
        print(round(max_threshold * 0.66, 0))

    def start(self, max_loop):
        count = 0

        # Changed the loop termination condition to check for convergence
        while count < max_loop:
            print("Loop " + str(count))
            self.last_cluster = [cluster.copy() for cluster in self.clusters]

            for j in range(len(self.clusters)):
                self.remove_points(j)
            for j in range(len(self.clusters)):
                self.check_point_change(j)

            if self.check_stabilize(self.last_cluster, self.clusters):
                self.plot_clusters(count)
                plt.show()

            self.new_centroid(self.clusters)

            print(self.clusters)

            # Check for convergence and stabilize
            if self.check_stabilize(self.last_cluster, self.clusters):
                print("STABILIZED")
                break

            count += 1

    def plot_clusters(self, count):
        plt.figure(count)
        print("Plotting")
        x = []
        cent_x = []
        y = []
        cent_y = []
        out_x = [x.x for x in self.outliers]
        out_y = [x.y for x in self.outliers]
        for j in range(len(self.clusters)):
            x.append(self.clusters[j].get_x_plot())
            cent_x.append([self.clusters[j].c.x])
            y.append(self.clusters[j].get_y_plot())
            cent_y.append([self.clusters[j].c.y])
        col = np.array([["blue"], ["orange"], ["green"]])
        for j in range(len(x)):
            plt.scatter(np.array(x[j]), np.array(y[j]), marker='.')
            plt.title("Loop: " + str(count))
        for j in range(len(cent_x)):
            plt.scatter(np.array(cent_x[j]), np.array(cent_y[j]), marker='*', c=col[j % len(col)])
            circle = plt.Circle((cent_x[j], cent_y[j]), self.max_threshold, fill=False)
            plt.gca().add_artist(circle)
            plt.gca().set_aspect(1)
        plt.scatter(np.array(out_x), np.array(out_y), marker='.', c="grey")

    def assign_points(self, p, c):
        tmp = []
        self.outliers = []
        for i in range(len(p)):
            for j in range(len(c)):
                if (self.check_distance(p[i], c[j].c)):
                    self.clusters[j].point_list.append(p[i])
                    tmp.append(p[i])
                    break
        self.outliers = [elem for elem in p if elem not in tmp]

    def generate_seed_points(self, p, cluster_count, max_threshold):
        p.sort()
        min_x = p[0].x
        min_y = 99999
        max_x = p[len(p) - 1].x
        max_y = -99999
        for i in p:
            if (i.y < min_y):
                min_y = i.y
            if (i.y > max_y):
                max_y = i.y
        self.clusters.append(Cluster(round(max_x * np.random.random_sample() + min_x, 2),
                                      round(max_y * np.random.random_sample() + min_y, 2), max_threshold))
        for i in range(cluster_count - 1):
            c = Cluster(round(max_x * np.random.random_sample() + min_x, 2),
                        round(max_y * np.random.random_sample() + min_y, 2), max_threshold)
            for j in range(len(self.clusters)):
                if (self.check_distance(c.c, self.clusters[j].c)):
                    c = Cluster(round(max_x * np.random.random_sample() + min_x, 2),
                                round(max_y * np.random.random_sample() + min_y, 2), max_threshold)
                    j = 0
            self.clusters.append(c)

    def new_centroid(self, c):
        for i in range(len(c)):
            new_x = sum([x.x for x in c[i].point_list])
            new_y = sum([x.y for x in c[i].point_list])
            if new_x != 0 or new_y != 0:
                new_x = new_x / len(c[i].point_list)
                new_y = new_y / len(c[i].point_list)
                c[i].c = Point((round(new_x, 2)), (round(new_y, 2)), 0)
        print("New Centroid: " + str(c))

    def remove_points(self, x):
        count = 0
        for i in range(len(self.clusters[x].point_list)):
            if not (self.check_distance(self.clusters[x].point_list[i - count], self.clusters[x].c)):
                print("Removed: " + str(self.clusters[x].point_list[i - count]) + " from: " + str(self.clusters[x].c))
                self.outliers.append(self.clusters[x].point_list[i - count])
                self.clusters[x].point_list.pop(i - count)
                count += 1

    def check_point_change(self, x):
        count = 0
        for i in range(len(self.outliers)):
            if i > len(self.outliers):
                break
            if (self.check_distance(self.outliers[i - count], self.clusters[x].c)):
                print("Added: " + str(self.outliers[i - count]) + " to: " + str(self.clusters[x].c))
                self.clusters[x].add_point(self.outliers[i - count])
                self.outliers.pop(i - count)
                count =+ 1

    def check_distance(self, p1, p2) -> bool:
        return (math.sqrt(((p1.x - p2.x) ** 2)) + math.sqrt(((p1.y - p2.y) ** 2)) <= self.max_threshold)

    def check_stabilize(self, c1, c2):
        if c1 == []:
            return False

        # Check if centroids moved less than the convergence thresh
        for i in range(len(c1)):
            if not (abs(c1[i].c.x - c2[i].c.x) < self.convergence_threshold and
                    abs(c1[i].c.y - c2[i].c.y) < self.convergence_threshold):
                return False
        return True

        

class Cluster():

    def __init__(self, x, y, radius):
        self.point_list = []
        self.c = Point(x, y, 0)
        self.radius = radius

    def center(self):
        return self.c

    def get_x_plot(self):
        return [x.x for x in self.point_list]

    def get_y_plot(self):
        return [x.y for x in self.point_list]

    def x(self):
        return self.c.x

    def y(self):
        return self.c.y

    def add_point(self, p):
        self.point_list.append(p)

    def copy(self):
        new_cluster = Cluster(self.c.x, self.c.y, self.radius)
        new_cluster.point_list = [point for point in self.point_list]
        return new_cluster

    def __repr__(self):
        return "(" + str(self.c.x) + ", " + str(self.c.y) + ")"

    def __str__(self) -> str:
        tmp = "Cluster: ["
        for i in self.point_list:
            tmp += (str(i) + ", ")
        tmp += "]"
        return tmp

    def __eq__(self, __o: object) -> bool:
        if self.c != __o.c:
            return False
        if (len(self.point_list) != len(__o.point_list)):
            return False
        self.point_list.sort()
        __o.point_list.sort()
        for i in range(len(self.point_list)):
            if (self.point_list[i] != __o.point_list[i]):
                return False
        return True

    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)

class Point():

    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id

    def __repr__(self) -> str:
        return "(" + str(self.x) + ", " + str(self.y) + ")"

    def __str__(self) -> str:
        return "(" + str(self.x) + "," + str(self.y) + ")"

    def __ne__(self, __o: object) -> bool:
        return (self.x, self.y)

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

    def __gt__(self, other):
        return (self.x, self.y) > (other.x, other.y)

    def __eq__(self, __o: object) -> bool:
        if self.x == __o.x and self.y == __o.y:
            return True
        else:
            return False

    def __ne__(self, __o: object) -> bool:
        return not self.__eq__(__o)

main()
