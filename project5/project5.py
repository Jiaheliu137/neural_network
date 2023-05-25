from utils import * 

import argparse

def main(cluster):
    x, label0, label_dict, I, P, color_dict = InputPattern("./iris.data")
    use_classes_in_iris = False
    use_wta_cluster = True if not use_classes_in_iris else False
    if use_wta_cluster:
        classes = np.loadtxt(f'./clusters_{cluster}.txt', dtype=str)
        label_dict = {label0[i]: classes[i] for i in range(len(classes))}

        unique_classes = np.unique(classes)
        color_list = ['red', 'blue', 'green', 'purple', 'cyan', 'orange', 'magenta', 'lime', 'navy', 'darkred', 'darkgreen', 'gold', 'teal']
        random.shuffle(color_list)  # Randomly shuffle the color list
        color_dict = {unique_classes[i]: color_list[i] for i in range(len(unique_classes))}
        print(f"color_dict:\n{color_dict}")

    w = np.random.random((N, I))
    w = SOFM(3000, 0.5, 0.04, 10, 1, w, x, I, P)
    # label = Calibration(w, x, label0, I, P)
    # print("\n\nResult after the first 1,000 iterations:\n")
    # PrintResult(label) 
    w = SOFM(7000, 0.04, 0.0, 1, 1, w, x, I, P)
    label, overlap_dict = Calibration(w, x, label0, I, P, label_dict)
    sorted_overlap_dict = sorted(overlap_dict.items(), key=lambda item: item[1])
    print(f"The labels assigned to the same neuron:\n")
    for k, v in sorted_overlap_dict:
        print(f"{k}: {v}")
    # print("\n\nResult after 20,000 iterations:\n")
    # PrintResult(label)
    PrintResult_figure(label, label_dict, 10000, color_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cluster", help="cluster number", default=3, type=int)
    args = parser.parse_args()
    main(args.cluster)
