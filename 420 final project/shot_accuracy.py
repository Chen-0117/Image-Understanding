from shot_detect import shot_detector

Movie1_true_boundary = [[0, 50], [51, 92], [93, 123], [124, 192], [193, 222], [223, 261], [262, 300], [301, 316],
                        [317, 339], [340, 391], [392, 412], [413, 429], [430, 454], [455, 456], [457, 514], [515, 552],
                        [553, 596], [597, 611], [612, 635], [636, 659], [660, 686], [687, 711], [712, 735], [736, 763],
                        [764, 791], [792, 805], [806, 825], [826, 845], [846, 870], [871, 886], [887, 912]]

Movie2_true_boundary = [[0, 51], [52, 77], [78, 109], [110, 149], [150, 211], [212, 250], [251, 278], [279, 296],
                        [297, 328], [329, 360], [361, 391], [392, 418], [419, 482], [483, 513], [514, 525], [526, 541],
                        [542, 566], [567, 585], [586, 615], [616, 641], [642, 693], [694, 728], [729, 757], [578, 776]]

Movie3_true_boundary = [[0, 4], [5, 45], [46, 80], [81, 156], [157, 191], [192, 233], [234, 274], [275, 315],
                        [316, 345], [346, 381], [382, 424], [425, 469], [470, 492], [493, 522], [523, 586]]


def accu(output, truth):
    out_put = []
    truth_ = []
    correct = 0
    miss = 0
    false = 0
    for i in range(1, len(output)):
        out_put.append(output[i][0])
    for i in range(1, len(truth)):
        truth_.append(truth[i][0])
    for i in truth_:
        if i in out_put:
            correct += 1
        else:
            miss += 1
    for i in out_put:
        if i not in truth_:
            false += 1
    return correct, miss, false


if __name__ == "__main__":
    o1 = shot_detector("Movie_1_new.mp4")
    o2 = shot_detector("Movie_2_new.mp4")
    o3 = shot_detector("Movie_3_new.mp4")
    print(accu(o1, Movie1_true_boundary))
    print(accu(o2, Movie2_true_boundary))
    print(accu(o3, Movie3_true_boundary))
