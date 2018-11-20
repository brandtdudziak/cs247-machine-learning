import matplotlib.pyplot as plt
import numpy as np
import math
from itertools import islice

if __name__ == "__main__":
    # Python2 handles integer division differently from Python3.
    # The following line will cause Python2 to terminate, but let Python3 continue.
    assert 1/2 == .5, "Use Python3, not Python2"

    # Create a square figure
    fig = plt.figure(figsize = (5, 5))

    # computes the distance between the given points
    def get_distance(point1, point2):
        sum = 0
        for index, x in enumerate(point1):
            sum += (x - point2[index])**2
        return math.sqrt(sum)

    # computes r/R for the given set of points
    def compute_qd(points):
        min_r = math.inf
        max_r = 0
        for index1, point1 in enumerate(points):
            for point2 in islice(points, index1+1, None):
                dist = get_distance(point1, point2)
                if(dist < min_r):
                    min_r = dist
                if(dist > max_r):
                    max_r = dist
        if(max_r == 0):
            return 0
        return min_r / max_r




    #Q1A
    #Parameters
    qd = []
    t = 5
    n = 10

    # Generate values for plot
    for d in range(1, 501):
        qd_sum = 0
        for i in range(t):
            # sample n points
            points = np.random.random_sample((n, d))
            # compute r/R
            qd_sum += compute_qd(points)
        qd.append(qd_sum / t)


    # Add q1a subfigure
    q1a = fig.add_subplot(1, 1, 1)
    q1a.set_xlabel(r"$d$")
    q1a.set_ylabel(r"$q(d)$")
    x = np.arange(1, 501)

    q1a.plot(x, qd)


    # Arrange everything in the plot such that we minimize overlap issues:
    plt.tight_layout()
    # Save the plot to disk, too:
    plt.savefig("q1a.pdf")




    #Q1B
    # Create a square figure
    fig = plt.figure(figsize = (5, 5))
    # Add q1a subfigure
    q1a = fig.add_subplot(1, 1, 1)
    q1a.set_xlabel(r"$d$")
    q1a.set_ylabel(r"$q(d)$")
    x = np.arange(1, 501)

    #Parameters
    t = 5

    # Generate values for plot
    for n in range(20, 51, 10):
        qd = []
        for d in range(1, 501):
            qd_sum = 0
            for i in range(t):
                # sample n points
                points = np.random.random_sample((n, d))
                # compute r/R
                qd_sum += compute_qd(points)
            qd.append(qd_sum / t)
        q1a.plot(x, qd)


    # Arrange everything in the plot such that we minimize overlap issues:
    plt.tight_layout()
    # Save the plot to disk, too:
    plt.savefig("q1b.pdf")



    # determines if a point is near an edge with epsilon 0.01
    def near_edge(point):
        for coord in point:
            if(coord <= 0.01 or coord >= 1 - 0.01):
                return 1
        return 0

    # determines if a point is within the hypersphere with radius 0.5
    def in_sphere(point):
        center = np.full(len(point), 0.5)
        dist = get_distance(point, center)
        if(dist <= 0.5):
            return 1
        return 0

    # computes the ratio of points that satisfies given condition
    def compute_ratio(points, total, function):
        tally = 0
        for point in points:
            tally += function(point)
        return tally / total


    #Q2A
    # Create a square figure
    fig = plt.figure(figsize = (5, 5))
    # Add q1a subfigure
    q1a = fig.add_subplot(1, 1, 1)
    q1a.set_xlabel(r"$d$")
    q1a.set_ylabel("Ratio of points near edge")
    x = np.arange(1, 501)

    #Parameters
    t = 10
    n = 100
    ratio = []

    # Generate values for plot
    for d in range(1, 501):
        ratio_sum = 0
        for i in range(t):
            # sample n points
            points = np.random.random_sample((n, d))
            # compute r/R
            ratio_sum += compute_ratio(points, n, near_edge)
        ratio.append(ratio_sum / t)

    q1a.plot(x, ratio)


    # Arrange everything in the plot such that we minimize overlap issues:
    plt.tight_layout()
    # Save the plot to disk, too:
    plt.savefig("q2a.pdf")


    #Q2B
    # Create a square figure
    fig = plt.figure(figsize = (5, 5))
    # Add q1a subfigure
    q1a = fig.add_subplot(1, 1, 1)
    q1a.set_xlabel(r"$d$")
    q1a.set_ylabel("Ratio of points contained in hypersphere")
    x = np.arange(1, 501)

    #Parameters
    t = 10
    n = 100
    ratio = []

    # Generate values for plot
    for d in range(1, 501):
        ratio_sum = 0
        for i in range(t):
            # sample n points
            points = np.random.random_sample((n, d))
            # compute r/R
            ratio_sum += compute_ratio(points, n, in_sphere)
        ratio.append(ratio_sum / t)

    q1a.plot(x, ratio)


    # Arrange everything in the plot such that we minimize overlap issues:
    plt.tight_layout()
    # Save the plot to disk, too:
    plt.savefig("q2b.pdf")
