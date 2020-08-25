import numpy as np
from time import time

def demo_loop(m, n):
    for i in range(n):
        print("hi")

    for j in range(m):
        for i in range(n):
            print("deep")


def demo_if(m, n):
    for j in range(m):
        for i in range(n):
            if j == 0:
                print("hi")
            print("deep")


def benchmark(fxn):
    all_durations = []
    for i in range(500):
        start = time()
        fxn(200, 5)
        duration = time() - start

        print(f"Run: {i} \t Duration: {duration}", )
        all_durations.append(duration)

    # print mean and std of durations.
    all_durations = np.array(all_durations)
    mean_time = np.mean(all_durations)
    std_time = np.std(all_durations)
    print(f"mean time: {mean_time} \t std time: {std_time}")


if __name__ == "__main__":
    # benchmark(demo_loop)
    benchmark(demo_if)

