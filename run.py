import itertools

from utilities import Features


def main():
    features = itertools.product(Features, repeat=len(Features))
    for feature in features:
        print(feature)


if __name__ == "__main__":
    main()