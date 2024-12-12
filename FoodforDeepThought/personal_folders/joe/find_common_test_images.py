from enum import member


def run():
    test_sets = {}

    with open('joe_test_set.txt', 'r') as f:
        source_set = [line.strip('\n') for line in f]

    with open('peter_test_set.txt', 'r') as f:
        test_sets['Peter'] = [line.strip('\n') for line in f]

    with open('prav_test_set.txt', 'r') as f:
        test_sets['Prav'] = [line.strip('\n') for line in f]

    print(source_set)
    print(test_sets['Peter'])

    common_set = []
    for image in source_set:
        in_common = True
        for member in test_sets:
            if image not in test_sets[member]:
                in_common = False

        if in_common:
            common_set.append(image)

    print(len(common_set))


if __name__ == "__main__":
    run()
