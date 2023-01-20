import numpy as np
from pathlib import Path
from os import path, remove
from scipy.io import wavfile
import glob
from code.ste_VAD.e_vad import one_vad_e
import matplotlib.pyplot as plt
from code.volume_VAD.SFE_VAD_t123 import vad1
import code.volume_VAD.Volume as vp
from code.Google_VAD import WebRTC_VAD
from code.lrt_VAD.LRT_VAD import main2
import code.lrt_VAD.LRT_estnoise_ms
from code.MFCC_VAD import MFCC_VAD
# from sklearn.metrics import auc


def score_labs(labPath, wavPath, SEG_LEN, SEG_STEP):
    """Function read .lab files
    Return array 'labs' with arrays ["fileName",[values]]
    values: '0' and '1' represents score for voiced and unvoiced segments
    """
    labs = []
    Paths = Path(labPath).glob("*.lab")
    count = 0
    for filePath in Paths:
        # print(filePath)
        fileName = path.splitext(path.basename(filePath))[0]
        # print("in file:" + fileName)
        labs.append([fileName, []])
        f_read = open(filePath, "r")
        lines = f_read.readlines()
        prev_end = 0
        # line_cnt = 0
        for line in lines:
            # print(line_cnt)
            # line_cnt += 1
            # add '0' from end of previous time stamp 'prev_end' to 'begin' beginning of new time stamp
            objects = line.split()
            begin = float(objects[0])
            end = float(objects[1])
            num_nuls = int(np.floor((begin - prev_end) / SEG_STEP)) if (begin - prev_end) % SEG_STEP < (SEG_STEP / 2) else int(np.ceil((begin - prev_end) / SEG_STEP))
            # if segDiff:
            #    print(segDiff)

            # add '1' from beginning of timestamp 'objects[0]' to 'objects[1]' end of time stamp
            num_ones = int(np.floor((end - begin) / SEG_STEP)) if (end - begin) % SEG_STEP < SEG_STEP / 2 else int(np.ceil((end - begin) / SEG_STEP))

            # if segments don't match actual number of segments at the end of time stamp, add difference to nuls
            segDiff = int(np.ceil((end / SEG_STEP) - (len(labs[count][1]) + num_nuls + num_ones)))
            # print("nuls: " + str(num_nuls))
            # print("ones: " + str(num_ones))
            # print("expected: " + str(int(np.ceil(end / SEG_STEP))) + " end: " + str(end))
            # print("real: " + str(len(labs[count][1]) + num_nuls + num_ones) + " diff: " + str(segDiff))
            # print("-------------------")
            if num_nuls + segDiff > 0:
                labs[count][1].extend(np.zeros(num_nuls + segDiff, dtype=int))
                labs[count][1].extend(np.ones(num_ones, dtype=int))
            else:
                labs[count][1].extend(np.zeros(num_nuls, dtype=int))
                labs[count][1].extend(np.ones(num_ones + segDiff, dtype=int))
            prev_end = end
            # print("ended line in file: " + fileName)

        f_read.close()
        # print(glob.glob("**/" + labs[count][0] + ".wav", recursive=True)[0])
        wavFile = glob.glob("**/" + labs[count][0] + ".wav", recursive=True)[0]
        sampleFreq, signal = wavfile.read(wavFile)
        signalLen = len(signal) / sampleFreq
        # if number of segments differs add zeros to the end
        # print(np.ceil(signalLen / SEG_STEP))
        # print(len(labs[count][1]))

        while np.ceil(signalLen / SEG_STEP) > len(labs[count][1]):
            labs[count][1].extend(np.zeros(1, dtype=int))
        count += 1
    return labs


def print_values(labs):
    """Print first lab file values
    """
    print(labs[0][1])


def test(wavPath, labs, SEG_STEP):
    """Test functionality of score_labs with original wav files
    Returns printed results
    """
    count = 0
    for lab in labs:
        NumOfSeg = len(labs[count][1])
        wavFile = glob.glob("**/" + labs[count][0] + ".wav", recursive=True)[0]
        sampleFreq, signal = wavfile.read(wavFile)  # wavPath + labs[count][0] + ".wav")
        signalLen = len(signal) / sampleFreq
        trueNumOfSeg = int(np.ceil(signalLen / SEG_STEP))
        # print(NumOfSeg)
        # print(trueNumOfSeg)
        if NumOfSeg != trueNumOfSeg:
            print("ERROR IN FILE: " + labs[count][0])
        else:
            print(labs[count][0] + ": DONE!")
        count += 1


def compare(orig, test):
    tp, tn, fp, fn = 0, 0, 0, 0
    # print(orig)
    # print(test)
    # exit()
    for i in range(len(orig)):
        orig_s = orig[i]
        test_s = test[i]
        if orig_s and test_s:
            tp += 1
        elif not orig_s and not test_s:
            tn += 1
        elif orig_s and not test_s:
            fn += 1
        else:
            fp += 1

    # print(tp, tn, fp, fn)
    return tp, tn, fp, fn


def comp_params(tp, tn, fp, fn):
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    TPR = sens
    FPR = 1 - spec
    return TPR, FPR


def make_roc(filename, wavpath, seg_len, seg_step, LAB_PATH):
    # th_arr = [0.01, 0.02, 0.03, 0.04 ,0.05, 0.06, 0.07 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.3]
    # th_arr = np.arange(0, 3, 0.1).tolist()
    # th_arr = [0, 2]

    th_arr = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 1, 2, 3, 4, 5, 10]  # th for energy vad
    # th_arr = [0, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]  # for LRT
    # th_arr = np.arange(0, 1, 0.1).tolist()
    lab = score_labs(LAB_PATH, wavpath, seg_len, seg_step)
    fprs = []
    tprs = []
    for th in th_arr:
        one_vad_e(wavpath, filename, th)  # energy VAD
        # main2(wavpath, filename, ".wav", th)  # LRT VAD
        test_lab = score_labs(".", wavpath, seg_len, seg_step)
        # print(len(lab[0][1]))
        # print(len(test_lab[0][1]))
        # exit()
        tp, tn, fp, fn = compare(lab[0][1], test_lab[0][1])
        # print(tp)
        # print(tn)
        # print(fp)
        # print(fn)
        tpr, fpr = comp_params(tp, tn, fp, fn)
        # print(tpr)
        # print(fpr)
        tprs.append(tpr)
        fprs.append(fpr)
    # tpr = []
    # for i in range(np.size(tprs)):
    #     tpr.append([tprs[i], fprs[i]])
    #     tpr.sort(key=lambda tup: tup[0])
    # tprs, fprs = [], []
    # for tp in tpr:
    #     tprs.append(tp[0])
    #     fprs.append(tp[1])
    return tprs, fprs


def plot_roc_curve(tpr, fpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of Energy VAD on HU PD_M')  # Receiver Operating Characteristic (ROC) Curve LRT VAD
    plt.legend()
    plt.show()


def make_auc_best(filename, wavpath, seg_len, seg_step, LAB_PATH):
    # th_arr = np.arange(0, 0.5, 0.001).tolist()
    th_arr = [0.015]
    # print("scoring original")
    lab = score_labs(LAB_PATH, wavpath, seg_len, seg_step)
    fprs = []
    tprs = []
    lst_err = 9999
    most_th = 0
    err_lst_rate = 1
    f = open("log.txt", "w+")
    for th in th_arr:
        # print("VAD test")
        one_vad_e(wavpath, filename, th)
        # print("scoring test")
        test_lab = score_labs(".", wavpath, seg_len, seg_step)
        print("scoring test ended")
        # print(len(lab[0][1]))
        # print(len(test_lab[0][1]))
        # exit()
        tp, tn, fp, fn = compare(lab[0][1], test_lab[0][1])
        tpr, fpr = comp_params(tp, tn, fp, fn)
        tprs.append(tpr)
        fprs.append(fpr)
        cnt_err, err_rate, t_cnt_err_000, t_cnt_err_001, t_cnt_err_011, t_cnt_err_010, t_cnt_err_100, t_cnt_err_101, t_cnt_err_110, t_cnt_err_111 = compare_best(lab[0][1], test_lab[0][1])
        f.write("TPR: {} FPR: {}\nThreshold: {}\nERR: {}\nERR_RATE: {}\nNDS: {}\nFEE: {}\nWC: {}\nFEC: {}\nOVER: {}\nWE: {}\nEXT: {}\nMSC: {}\n\n".format(tpr, fpr, th, cnt_err, err_rate, t_cnt_err_000, t_cnt_err_001, t_cnt_err_011, t_cnt_err_010, t_cnt_err_100, t_cnt_err_101, t_cnt_err_110, t_cnt_err_111))
        if cnt_err < lst_err:
            cnt_err_000, cnt_err_001, cnt_err_011, cnt_err_010, cnt_err_100, cnt_err_101, cnt_err_110, cnt_err_111 = t_cnt_err_000, t_cnt_err_001, t_cnt_err_011, t_cnt_err_010, t_cnt_err_100, t_cnt_err_101, t_cnt_err_110, t_cnt_err_111
            most_th = th
            lst_err = cnt_err
            x, y = fpr, tpr
            err_lst_rate = err_rate
        # remove("./" + filename + ".lab")
    # auc_score = auc(fprs, tprs)
    disp_err(x, y, most_th, lst_err, err_lst_rate, cnt_err_000, cnt_err_001, cnt_err_011, cnt_err_010, cnt_err_100, cnt_err_101, cnt_err_110, cnt_err_111)
    f.close()


def make_auc_best_SFE_VAD(filename, wavpath, seg_len, seg_step, LAB_PATH):
    # th_arr = np.arange(0, 1, 0.001).tolist()
    th_arr = [0.108]
    lab = score_labs(LAB_PATH, wavpath, seg_len, seg_step)
    fprs = []
    cnt_err_000, cnt_err_001, cnt_err_011, cnt_err_010, cnt_err_100, cnt_err_101, cnt_err_110, cnt_err_111 = 0, 0, 0, 0, 0, 0, 0, 0
    tprs = []
    lst_err = 9999
    most_th = 0
    err_lst_rate = 1
    f = open("log.txt", "w+")
    x, y = 0, 0
    for th in th_arr:
        # print("here")
        vad1(wavpath, filename, ".wav", th, 3)
        # print("here")
        test_lab = score_labs(".", wavpath, seg_len, seg_step)
        # print("here")
        # print(len(lab[0][1]))
        # print(len(test_lab[0][1]))
        # exit()
        tp, tn, fp, fn = compare(lab[0][1], test_lab[0][1])
        # print("here")
        tpr, fpr = comp_params(tp, tn, fp, fn)
        # print("here")
        tprs.append(tpr)
        fprs.append(fpr)
        cnt_err, err_rate, t_cnt_err_000, t_cnt_err_001, t_cnt_err_011, t_cnt_err_010, t_cnt_err_100, t_cnt_err_101, t_cnt_err_110, t_cnt_err_111 = compare_best(lab[0][1], test_lab[0][1])
        f.write("TPR: {} FPR: {}\nThreshold: {}\nERR: {}\nERR_RATE: {}\nNDS: {}\nFEE: {}\nWC: {}\nFEC: {}\nOVER: {}\nWE: {}\nEXT: {}\nMSC: {}\n\n".format(tpr, fpr, th, cnt_err, err_rate, t_cnt_err_000, t_cnt_err_001, t_cnt_err_011, t_cnt_err_010, t_cnt_err_100, t_cnt_err_101, t_cnt_err_110, t_cnt_err_111))
        # print("here")
        if cnt_err < lst_err:
            cnt_err_000, cnt_err_001, cnt_err_011, cnt_err_010, cnt_err_100, cnt_err_101, cnt_err_110, cnt_err_111 = t_cnt_err_000, t_cnt_err_001, t_cnt_err_011, t_cnt_err_010, t_cnt_err_100, t_cnt_err_101, t_cnt_err_110, t_cnt_err_111
            most_th = th
            lst_err = cnt_err
            x, y = fpr, tpr
            err_lst_rate = err_rate
        print("one th ended")
    # auc_score = auc(fprs, tprs)
    disp_err(x, y, most_th, lst_err, err_lst_rate, cnt_err_000, cnt_err_001, cnt_err_011, cnt_err_010, cnt_err_100, cnt_err_101, cnt_err_110, cnt_err_111)
    f.close()


def make_auc_best_MFCC(filename, wavpath, seg_len, seg_step, LAB_PATH):
    th = "none"
    lab = score_labs(LAB_PATH, wavpath, seg_len, seg_step)
    fprs = []
    tprs = []
    lst_err = 9999
    most_th = 0
    err_lst_rate = 1
    f = open("log.txt", "w+")
    MFCC_VAD.main("")
    test_lab = score_labs(".", wavpath, seg_len, seg_step)
    # print(len(lab[0][1]))
    # print(len(test_lab[0][1]))
    # exit()
    tp, tn, fp, fn = compare(lab[0][1], test_lab[0][1])
    tpr, fpr = comp_params(tp, tn, fp, fn)
    tprs.append(tpr)
    fprs.append(fpr)
    cnt_err, err_rate, t_cnt_err_000, t_cnt_err_001, t_cnt_err_011, t_cnt_err_010, t_cnt_err_100, t_cnt_err_101, t_cnt_err_110, t_cnt_err_111 = compare_best(lab[0][1], test_lab[0][1])
    f.write("TPR: {} FPR: {}\nThreshold: {}\nERR: {}\nERR_RATE: {}\nNDS: {}\nFEE: {}\nWC: {}\nFEC: {}\nOVER: {}\nWE: {}\nEXT: {}\nMSC: {}\n\n".format(tpr, fpr, th, cnt_err, err_rate, t_cnt_err_000, t_cnt_err_001, t_cnt_err_011, t_cnt_err_010, t_cnt_err_100, t_cnt_err_101, t_cnt_err_110, t_cnt_err_111))
    if cnt_err < lst_err:
        cnt_err_000, cnt_err_001, cnt_err_011, cnt_err_010, cnt_err_100, cnt_err_101, cnt_err_110, cnt_err_111 = t_cnt_err_000, t_cnt_err_001, t_cnt_err_011, t_cnt_err_010, t_cnt_err_100, t_cnt_err_101, t_cnt_err_110, t_cnt_err_111
        most_th = th
        lst_err = cnt_err
        x, y = fpr, tpr
        err_lst_rate = err_rate
    # auc_score = auc(fprs, tprs)
    disp_err(x, y, most_th, lst_err, err_lst_rate, cnt_err_000, cnt_err_001, cnt_err_011, cnt_err_010, cnt_err_100, cnt_err_101, cnt_err_110, cnt_err_111)
    f.close()


def make_auc_best_rtc(filename, wavpath, seg_len, seg_step, LAB_PATH, mode, fl):
    lab = score_labs(LAB_PATH, wavpath, seg_len, seg_step)
    fprs = []
    tprs = []
    lst_err = 9999
    th = "mode x"
    err_lst_rate = 1
    f = open("log.txt", "w+")
    WebRTC_VAD.main2(wavpath, filename, ".wav", mode, fl)
    test_lab = score_labs(".", wavpath, seg_len, seg_step)
    # print(len(lab[0][1]))
    # print(len(test_lab[0][1]))
    # exit()
    tp, tn, fp, fn = compare(lab[0][1], test_lab[0][1])
    tpr, fpr = comp_params(tp, tn, fp, fn)
    tprs.append(tpr)
    fprs.append(fpr)
    cnt_err, err_rate, t_cnt_err_000, t_cnt_err_001, t_cnt_err_011, t_cnt_err_010, t_cnt_err_100, t_cnt_err_101, t_cnt_err_110, t_cnt_err_111 = compare_best(lab[0][1], test_lab[0][1])
    f.write("TPR: {} FPR: {}\nThreshold: {}\nERR: {}\nERR_RATE: {}\nNDS: {}\nFEE: {}\nWC: {}\nFEC: {}\nOVER: {}\nWE: {}\nEXT: {}\nMSC: {}\n\n".format(tpr, fpr, th, cnt_err, err_rate, t_cnt_err_000, t_cnt_err_001, t_cnt_err_011, t_cnt_err_010, t_cnt_err_100, t_cnt_err_101, t_cnt_err_110, t_cnt_err_111))
    if cnt_err < lst_err:
        cnt_err_000, cnt_err_001, cnt_err_011, cnt_err_010, cnt_err_100, cnt_err_101, cnt_err_110, cnt_err_111 = t_cnt_err_000, t_cnt_err_001, t_cnt_err_011, t_cnt_err_010, t_cnt_err_100, t_cnt_err_101, t_cnt_err_110, t_cnt_err_111
        most_th = th
        lst_err = cnt_err
        x, y = fpr, tpr
        err_lst_rate = err_rate
    # auc_score = auc(fprs, tprs)
    disp_err(x, y, most_th, lst_err, err_lst_rate, cnt_err_000, cnt_err_001, cnt_err_011, cnt_err_010, cnt_err_100, cnt_err_101, cnt_err_110, cnt_err_111)
    f.close()


def make_auc_best_LRT(filename, wavpath, seg_len, seg_step, LAB_PATH):
    th_arr = np.arange(0, 1, 0.001).tolist()
    # th_arr = [0]
    lab = score_labs(LAB_PATH, wavpath, seg_len, seg_step)
    fprs = []
    tprs = []
    lst_err = 9999
    most_th = 0
    err_lst_rate = 1
    f = open("log.txt", "w+")
    for th in th_arr:
        main2(wavpath, filename, ".wav", th)
        test_lab = score_labs(".", wavpath, seg_len, seg_step)
        # print(len(lab[0][1]))
        # print(len(test_lab[0][1]))
        # exit()
        tp, tn, fp, fn = compare(lab[0][1], test_lab[0][1])
        tpr, fpr = comp_params(tp, tn, fp, fn)
        tprs.append(tpr)
        fprs.append(fpr)
        cnt_err, err_rate, t_cnt_err_000, t_cnt_err_001, t_cnt_err_011, t_cnt_err_010, t_cnt_err_100, t_cnt_err_101, t_cnt_err_110, t_cnt_err_111 = compare_best(lab[0][1], test_lab[0][1])
        f.write("TPR: {} FPR: {}\nThreshold: {}\nERR: {}\nERR_RATE: {}\nNDS: {}\nFEE: {}\nWC: {}\nFEC: {}\nOVER: {}\nWE: {}\nEXT: {}\nMSC: {}\n\n".format(tpr, fpr, th, cnt_err, err_rate, t_cnt_err_000, t_cnt_err_001, t_cnt_err_011, t_cnt_err_010, t_cnt_err_100, t_cnt_err_101, t_cnt_err_110, t_cnt_err_111))
        if cnt_err <= lst_err:
            cnt_err_000, cnt_err_001, cnt_err_011, cnt_err_010, cnt_err_100, cnt_err_101, cnt_err_110, cnt_err_111 = t_cnt_err_000, t_cnt_err_001, t_cnt_err_011, t_cnt_err_010, t_cnt_err_100, t_cnt_err_101, t_cnt_err_110, t_cnt_err_111
            most_th = th
            lst_err = cnt_err
            x, y = fpr, tpr
            err_lst_rate = err_rate
    # auc_score = auc(fprs, tprs)
    disp_err(x, y, most_th, lst_err, err_lst_rate, cnt_err_000, cnt_err_001, cnt_err_011, cnt_err_010, cnt_err_100, cnt_err_101, cnt_err_110, cnt_err_111)
    f.close()


def compare_best(orig, test):
    prev = 5
    cnt, cnt_err, err_000, err_001, err_011, err_010, err_100, err_101, err_110, err_111 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cnt_err_000, cnt_err_001, cnt_err_011, cnt_err_010, cnt_err_100, cnt_err_101, cnt_err_110, cnt_err_111 = 0, 0, 0, 0, 0, 0, 0, 0
    # print(len(orig))
    # print(len(test))
    # exit()
    for i in range(len(orig)):
        cnt += 1
        orig_s = orig[i]
        test_s = test[i]
        if orig_s and test_s:
            if err_001:
                cnt_err_001 += 1
                err_000, err_001, err_011, err_010, err_100, err_101, err_110, err_111 = 0, 0, 0, 0, 0, 0, 0, 0
            elif err_011:
                cnt_err_011 += 1
                err_000, err_001, err_011, err_010, err_100, err_101, err_110, err_111 = 0, 0, 0, 0, 0, 0, 0, 0
            elif err_101:
                cnt_err_101 += 1
                err_000, err_001, err_011, err_010, err_100, err_101, err_110, err_111 = 0, 0, 0, 0, 0, 0, 0, 0
            elif err_111:
                cnt_err_111 += 1
                err_000, err_001, err_011, err_010, err_100, err_101, err_110, err_111 = 0, 0, 0, 0, 0, 0, 0, 0
            prev = 1
        elif not orig_s and not test_s:
            if err_000:
                cnt_err_000 += 1
                err_000, err_001, err_011, err_010, err_100, err_101, err_110, err_111 = 0, 0, 0, 0, 0, 0, 0, 0
            elif err_010:
                cnt_err_010 += 1
                err_000, err_001, err_011, err_010, err_100, err_101, err_110, err_111 = 0, 0, 0, 0, 0, 0, 0, 0
            elif err_100:
                cnt_err_100 += 1
                err_000, err_001, err_011, err_010, err_100, err_101, err_110, err_111 = 0, 0, 0, 0, 0, 0, 0, 0
            elif err_110:
                cnt_err_110 += 1
                err_000, err_001, err_011, err_010, err_100, err_101, err_110, err_111 = 0, 0, 0, 0, 0, 0, 0, 0
            prev = 0
        elif orig_s and not test_s:
            cnt_err += 1
            if prev == 5:
                pass
            elif prev == 0:
                err_011 = 1
                err_010 = 1
            elif prev == 1:
                err_110 = 1
                err_111 = 1
            prev = 1
        else:
            cnt_err += 1
            if prev == 5:
                pass
            elif prev == 0:
                err_000 = 1
                err_001 = 1
            elif prev == 1:
                err_100 = 1
                err_101 = 1
            prev = 0

    # print(tp, tn, fp, fn)
    err_rate = cnt_err / cnt
    return cnt_err, err_rate, cnt_err_000, cnt_err_001, cnt_err_011, cnt_err_010, cnt_err_100, cnt_err_101, cnt_err_110, cnt_err_111


def disp_err(y, x, th, err, err_rate, cnt_err_000, cnt_err_001, cnt_err_011, cnt_err_010, cnt_err_100, cnt_err_101, cnt_err_110, cnt_err_111):
    # print("NDS: %d\n FEE: %d\n", cnt_err_000, cnt_err_001)
    f = open("logsum.txt", "a+")
    f.write("TPR: {} FPR: {}\nThreshold: {}\nERR: {}\nERR_RATE: {}\nNDS: {}\nFEE: {}\nWC: {}\nFEC: {}\nOVER: {}\nWE: {}\nEXT: {}\nMSC: {}\n\n".format(x, y, th, err, err_rate, cnt_err_000, cnt_err_001, cnt_err_011, cnt_err_010, cnt_err_100, cnt_err_101, cnt_err_110, cnt_err_111))
    f.close()
    print("TPR: {}   FPR: {}\nThreshold: {}\nERR: {}\nNDS: {}\nFEE: {}\nWC: {}\nFEC: {}\nOVER: {}\nWE: {}\nEXT: {}\nMSC: {}\n".format(y, x, th, err, cnt_err_000, cnt_err_001, cnt_err_011, cnt_err_010, cnt_err_100, cnt_err_101, cnt_err_110, cnt_err_111))


if __name__ == '__main__':
    """Program takes lab files for HC_F_01_TSK7.wav a computes binary values
    of voiced and unvoiced segments
    Result is saved in 'labs' and 'test_labs'
    """
    LAB_PATH = "vysledky/"  # path to original lab files
    # TEST_PATH = "vysledky/VAD_E/"  # path to lab files to be scored
    WAV_PATH = "/home/komplike/bp/nahravky/drive/DDK/"  # path to TSK7 directory where wav files are

    SEG_LEN = 0.025  # s
    SEG_STEP = 0.015  # s
    # lab = score_labs(LAB_PATH, WAV_PATH, SEG_LEN, SEG_STEP)
    # test_lab = score_labs(TEST_PATH, WAV_PATH, SEG_LEN, SEG_STEP)
    # test(WAV_PATH, test_labs, SEG_STEP)
    filename = "PD_F_TSK7"
    # tpr, fpr = make_roc(filename, WAV_PATH, SEG_LEN, SEG_STEP, LAB_PATH)
    # print(tpr)
    # print(fpr)
    # plot_roc_curve(tpr, fpr)
    # fpr.sort()
    # plot_roc_curve(tpr, fpr)
    # make_auc_best(filename, WAV_PATH, SEG_LEN, SEG_STEP, LAB_PATH)
    # remove("./" + filename + ".lab")
    make_auc_best_SFE_VAD(filename, WAV_PATH, SEG_LEN, SEG_STEP, LAB_PATH)
    # remove("./" + filename + ".lab")
    # make_auc_best_MFCC(filename, WAV_PATH, SEG_LEN, SEG_STEP, LAB_PATH)
    # remove("./" + filename + ".lab")
    # for mode in range(4):
    #     for fl in range(3):
    #         print(mode)
    #         print((fl + 1) * 10)
    #         make_auc_best_rtc(filename, WAV_PATH, SEG_LEN, SEG_STEP, LAB_PATH, mode, (fl + 1) * 10)
    # remove("./" + filename + ".lab")
    # make_auc_best_LRT(filename, WAV_PATH, SEG_LEN, SEG_STEP, LAB_PATH)
    # remove("./" + filename + ".lab")

    LAB_PATH = "vysledky/t_pd_m"
    filename = "PD_M_TSK7"
    # make_auc_best(filename, WAV_PATH, SEG_LEN, SEG_STEP, LAB_PATH)
    # remove("./" + filename + ".lab")
    # make_auc_best_SFE_VAD(filename, WAV_PATH, SEG_LEN, SEG_STEP, LAB_PATH)
    # remove("./" + filename + ".lab")
    # make_auc_best_MFCC(filename, WAV_PATH, SEG_LEN, SEG_STEP, LAB_PATH)
    # remove("./" + filename + ".lab")
    # for mode in range(4):
    #     for fl in range(3):
    #         print(mode)
    #         print((fl + 1) * 10)
    #         make_auc_best_rtc(filename, WAV_PATH, SEG_LEN, SEG_STEP, LAB_PATH, mode, (fl + 1) * 10)
    # remove("./" + filename + ".lab")
    # make_auc_best_LRT(filename, WAV_PATH, SEG_LEN, SEG_STEP, LAB_PATH)
    # remove("./" + filename + ".lab")

    LAB_PATH = "vysledky/t_pd_f"
    filename = "PD_F_TSK7"
    # make_auc_best(filename, WAV_PATH, SEG_LEN, SEG_STEP, LAB_PATH)
    # remove("./" + filename + ".lab")
    # make_auc_best_SFE_VAD(filename, WAV_PATH, SEG_LEN, SEG_STEP, LAB_PATH)
    # remove("./" + filename + ".lab")
    # make_auc_best_MFCC(filename, WAV_PATH, SEG_LEN, SEG_STEP, LAB_PATH)
    # remove("./" + filename + ".lab")
    # for mode in range(4):
    #     for fl in range(3):
    #         print(mode)
    #         print((fl + 1) * 10)
    #         make_auc_best_rtc(filename, WAV_PATH, SEG_LEN, SEG_STEP, LAB_PATH, mode, (fl + 1) * 10)
    # remove("./" + filename + ".lab")
    # make_auc_best_LRT(filename, WAV_PATH, SEG_LEN, SEG_STEP, LAB_PATH)
    # remove("./" + filename + ".lab")
