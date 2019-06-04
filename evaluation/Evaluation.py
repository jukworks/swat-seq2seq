import sys, getopt
import time, datetime

class Single_Period:
    def __init__(self, first, last, name):
        self._first_timestamp = first
        self._last_timestamp = last
        self._name = name

    def set_time(self, first, last):
        self._first_timestamp = first
        self._last_timestamp = last

    def get_time(self):
        return self._first_timestamp, self._last_timestamp

    def set_name(self, str):
        self._name = str

    def get_name(self):
        return self._name

    def __eq__(self, other):
        return self._first_timestamp == other.get_time()[0] and self._last_timestamp == other.get_time()[1]


class TimeConvertor:
    def __init__(self):
        pass

    def unixtime_to_string(self, unixtime):
        return datetime.datetime.fromtimestamp(unixtime).strftime('%Y-%m-%d %I:%M:%S %p')

    def string_to_unixtime(self, str):
        return time.mktime(datetime.datetime.strptime(str, " %d/%m/%Y %I:%M:%S %p").timetuple())

    def index_to_string(self, idx):
        if idx >= 298837:
            return self.unixtime_to_string(idx + 1451264400 + 81)
        else:
            return self.unixtime_to_string(idx + 1451264400)


class Evaluation:
    def __init__(self):
        self._prediction = [] #list
        self._ground_truth = [] #list

        self._set_prediction = False
        self._set_ground_truth = False
        pass

    def set_prediction(self, prediction):
        self._prediction = self._head_tail_index(prediction)
        self._set_prediction = True

    def _head_tail_index(self, prediction):
        return_list = []
        start_idx = -1

        if prediction[0] == 1:
            start_idx = 0

        for idx in range(1, len(prediction)):
            if prediction[idx] == 1 and prediction[idx - 1] == 0:
                start_idx = idx

            if prediction[idx] == 0 and prediction[idx - 1] == 1:
                return_list.append([start_idx, idx - 1])

        if prediction[len(prediction) - 1] == 1:
            return_list.append([start_idx, len(prediction) - 1])

        return return_list


    def get_n_predictions(self):
        return len(self._prediction)

    def set_ground_truth(self, ground_truth_list):
        self._ground_truth = []
        for single_answer in ground_truth_list:
            self._ground_truth.append(Single_Period(single_answer[0], single_answer[1], str(single_answer[2])))

        self._set_ground_truth = True


    def _extend_single_truth(self, single_list, extend_head, extend_tail):
        for idx in range(len(single_list)):
            if idx != 0:
                first = max(single_list[idx].get_time()[0] - extend_head, single_list[idx-1].get_time()[1])
            else:
                first = single_list[idx].get_time()[0] - extend_head

            if first < 0:
                first = 0

            if idx != len(single_list)-1:
                last = min(single_list[idx].get_time()[1] + extend_tail, single_list[idx+1].get_time()[0])
            else:
                last = single_list[idx].get_time()[1] + extend_tail
            single_list[idx].set_time(first, last)


    def extend_ground_truth(self, extend_head, extend_tail):
        self._extend_single_truth(self._ground_truth, extend_head, extend_tail)


    def _is_intersection(self, prediction_term, attack_term):
        if prediction_term[1] >= attack_term[0] and attack_term[1] >= prediction_term[0]:
            return True
        else:
            return False


    def _find_hit_lists(self, intersect_ratio, multi_hit=True):
        hit_ground_truth = []
        hit_prediction = []

        for idx in range(len(self._ground_truth)):
            hit_ground_truth.append(False)
        for idx in range(len(self._prediction)):
            hit_prediction.append(False)

        for idx_gt in range(len(self._ground_truth)):
            for idx_pr in range(len(self._prediction)):
                length = self._intersect_len(self._ground_truth[idx_gt], self._prediction[idx_pr])
                term_len = self._prediction[idx_pr][1] - self._prediction[idx_pr][0] + 1
                if float(length/term_len) >= intersect_ratio:
                    if multi_hit == True or hit_prediction[idx_pr] == False:
                        hit_ground_truth[idx_gt] = True
                    hit_prediction[idx_pr] = True

        return hit_prediction, hit_ground_truth


    def detected_attacks(self, intersect_ratio, multi_hit=True):
        assert (self._set_ground_truth == True and self._set_prediction == True)

        hit_ground_truth = []
        hit_prediction = []

        for idx in range(len(self._ground_truth)):
            hit_ground_truth.append(False)
        for idx in range(len(self._prediction)):
            hit_prediction.append(False)

        for idx_gt in range(len(self._ground_truth)):
            for idx_pr in range(len(self._prediction)):
                length = self._intersect_len(self._ground_truth[idx_gt].get_time(), self._prediction[idx_pr])
                term_len = self._prediction[idx_pr][1] - self._prediction[idx_pr][0] + 1
                if float(length/term_len) >= intersect_ratio:
                    if multi_hit == True or hit_prediction[idx_pr] == False:
                        hit_ground_truth[idx_gt] = True
                    hit_prediction[idx_pr] = True

        found_cnt = 0
        found_str = ' - Found: '
        notfound_str = ' - Not Found: '
        for idx in range(len(hit_ground_truth)):
            if hit_ground_truth[idx] == True:
                found_str += self._ground_truth[idx].get_name() + ', '
                found_cnt += 1
            else:
                notfound_str += self._ground_truth[idx].get_name() + ', '

        print('#Found attacks: ' + str(found_cnt) + '/' + str(len(self._ground_truth)))
        print('Detecting Ratio: ' + str(round(float(found_cnt)/len(self._ground_truth), 2)))
        print(found_str[:-2])
        print(notfound_str[:-2])


    def false_alarm(self, intersect_ratio, multi_hit=True):
        assert (self._set_ground_truth == True and self._set_prediction == True)

        tc = TimeConvertor()
        fa_list = []

        cnt = 0;
        sum = 0.0;
        for prediction_term in self._prediction:
            non_attack_time = self._non_attack_time(prediction_term, self._ground_truth, multi_hit)
            if 1-float(non_attack_time)/(prediction_term[1] - prediction_term[0] + 1) <= intersect_ratio:
                fa_list.append([tc.index_to_string(prediction_term[0]), tc.index_to_string(prediction_term[1])])
                cnt += 1
                sum += non_attack_time

        #print('[False alarm]')
        print('#False alarm: ' + str(cnt) + '/' + str(len(self._prediction)))
        if cnt == 0:
            print(' - Total (Average) time: 0 (0)')
        else:
            print(' - Total (Average) time: ' + str(sum) + ' (' + str(round(sum / cnt,2)) + ')')
        for fa in fa_list:
            print('   ' + fa[0] + ' ~ ' + fa[1])


    def _intersect_len(self, index1, index2):
        if index1[1] < index2[0] or index2[1] < index1[0]:
            return -1  # no intersection

        if index1[0] < index2[0]:
            if index1[1] < index2[1]:
                return index1[1] - index2[0] + 1
            else:
                return index2[1] - index2[0] + 1
        else:
            if index1[1] < index2[1]:
                return index1[1] - index1[0] + 1
            else:
                return index2[1] - index1[0] + 1

    def _non_attack_time(self, detect_term, ground_truth, multi_hit):
        non_attack_time = detect_term[1] - detect_term[0] + 1

        for idx in range(len(ground_truth)):
            attack_term = ground_truth[idx].get_time()
            coocurrence = self._intersect_len(detect_term, attack_term)
            if coocurrence != -1:
                non_attack_time -= coocurrence
                if multi_hit == False:
                    break

        return non_attack_time


#Process별로 평가하기 위해 필요
#i번째 프로세스에 대한 공격은 i번째 프로세스에 대한 예측 뿐만 아니라 i+1번째 예측에 포함될 수 있음
#i번째 프로세스에 대한 예측이 false alarm인지를 평가하기 위해서는 i-1번째 공격 데이터를 확인해야 함
class Evaluation_SWaT_Process(Evaluation):
    def __init__(self):
        Evaluation.__init__(self)
        pass


    def set_prediction(self, prediction_lists):
        for idx in range(len(prediction_lists)):
            self._prediction.append(self._head_tail_index(prediction_lists[idx]))

        self._set_prediction = True


    def set_ground_truth(self, ground_truth_lists):
        self._ground_truth = []
        list_idx = 0

        for single_list in ground_truth_lists:
            self._ground_truth.append([])
            for single_answer in single_list:
                self._ground_truth[list_idx].append(Single_Period(single_answer[0], single_answer[1], str(single_answer[2])))
            list_idx += 1
        self._set_ground_truth = True


    def extend_ground_truth(self, extend_head, extend_tail):
        for single_list in self._ground_truth:
            self._extend_single_truth(single_list, extend_head, extend_tail)

    def get_n_predictions(self, idx):
        return len(self._prediction[idx])

    def _detected_attack_2lists(self, gt_pid, prd_pid, intersect_ratio, multi_hit):
        ground_truth = self._ground_truth[gt_pid]
        prediction = self._prediction[prd_pid]

        hit_ground_truth = []
        hit_prediction = []

        for idx in range(len(ground_truth)):
            hit_ground_truth.append(False)
        for idx in range(len(prediction)):
            hit_prediction.append(False)

        for idx_gt in range(len(ground_truth)):
            for idx_pr in range(len(prediction)):
                length = self._intersect_len(ground_truth[idx_gt].get_time(), prediction[idx_pr])
                term_len = prediction[idx_pr][1] - prediction[idx_pr][0] + 1
                if float(length / term_len) >= intersect_ratio:
                    if multi_hit == True or hit_prediction[idx_pr] == False:
                        hit_ground_truth[idx_gt] = True
                    hit_prediction[idx_pr] = True

        return hit_ground_truth

    def detected_attacks(self, pid, preceding_len, intersect_ratio, multi_hit=True):
        assert (self._set_ground_truth == True and self._set_prediction == True)
        assert (pid >= 0 and pid < len(self._ground_truth))

        print_buf_list = []
        prev_hit_list = [False] * len(self._ground_truth[pid])
        for i in range(pid, min(pid+preceding_len+1, len(self._prediction))):
            print_buf = ''
            hit_list = self._detected_attack_2lists(pid, i, intersect_ratio, multi_hit)
            print_buf += ' - Found at Process #' + str(i) + ': '
            for j in range(len(prev_hit_list)):
                if hit_list[j] == True and prev_hit_list[j] == False:
                    prev_hit_list[j] = True
                    print_buf += self._ground_truth[pid][j].get_name() + ', '
            print_buf_list.append(print_buf)

        found_cnt = 0
        notfound_str = ' - Not Found: '
        for idx in range(len(prev_hit_list)):
            if prev_hit_list[idx] == True:
                found_cnt += 1
            else:
                notfound_str += self._ground_truth[pid][idx].get_name() + ', '

        print('#Found attacks: ' + str(found_cnt) + '/' + str(len(self._ground_truth[pid])))
        print(' - Detecting Ratio: ' + str(round(float(found_cnt) / len(self._ground_truth[pid]), 2)))
        for line in print_buf_list:
            if len(line) != 24:
                print(line[:-2])
        print(notfound_str[:-2])


    def _false_alarm_2lists(self, gt_pid, prd_pid, non_attack_times, multi_hit):
        prediction = self._prediction[prd_pid]

        #process간 중복되는 공격 리스트 제거
        ground_truth = []
        for term in self._ground_truth[gt_pid]:
            exist = False
            for idx in range(gt_pid+1, prd_pid+1):
                for term2 in self._ground_truth[idx]:
                    if term == term2:
                        exist = True
                        break
                if exist == True:
                    break
            if exist == False:
                ground_truth.append(term)

        for idx in range(len(prediction)):
            org_term = prediction[idx][1] - prediction[idx][0] + 1
            if multi_hit == True or org_term == non_attack_times[idx]:
                non_attack_times[idx] -= org_term - self._non_attack_time(prediction[idx], ground_truth, multi_hit)


    def false_alarm(self, pid, preceding_len, intersect_ratio, multi_hit=True):
        assert (self._set_ground_truth == True and self._set_prediction == True)

        fa_list = []
        tc = TimeConvertor()

        cnt = 0;
        sum = 0.0;
        non_attack_times = []
        for prediction_term in self._prediction[pid]:
            val = prediction_term[1] - prediction_term[0] + 1
            non_attack_times.append(prediction_term[1] - prediction_term[0] + 1)

        for idx in range(pid, max(-1, pid-preceding_len-1),-1):
            self._false_alarm_2lists(idx, pid, non_attack_times, multi_hit)

        for idx in range(len(self._prediction[pid])):
            if 1 - float(non_attack_times[idx]) / (self._prediction[pid][idx][1] - self._prediction[pid][idx][0] + 1) <= intersect_ratio:
                fa_list.append([tc.index_to_string(self._prediction[pid][idx][0]), tc.index_to_string(self._prediction[pid][idx][1])])
                cnt += 1
                sum += non_attack_times[idx]

        #print('[False alarm]')
        print('#False alarm: ' + str(cnt) + '/' + str(len(self._prediction[pid])))
        if cnt == 0:
            print(' - Total (Average) time: 0 (0)')
        else:
            print(' - Total (Average) time: ' + str(sum) + ' (' + str(round(sum / cnt,2)) + ')')
        for fa in fa_list:
            print('   ' + fa[0] + ' ~ ' + fa[1])



def read_timeseries_file(filename):
    rbuf = []

    f = open(filename, 'r', encoding='utf-8', newline='')
    for line in f.readlines():
        items = line.strip().split(',')
        rbuf.append(int(i) for i in items)
    f.close()

    return list(map(list, zip(*rbuf)))


def read_period_file(filename):
    rbuf = []

    f = open(filename, 'r', encoding='utf-8', newline='')
    for line in f.readlines():
        tup = []
        items = line.strip().split(',')
        #rbuf.append(list(int(i) for i in items))
        tup.append(int(items[0]))
        tup.append(int(items[1]))
        tup.append(str(items[2]))
        rbuf.append(tup)
    f.close()

    return list(rbuf)


def main(argv):
    directory = './Attack_Data_Merging'
    attack_file = [ directory + '/SWaT_Attack_all.csv',
                 directory + '/SWaT_Attack_pid0.csv',
                 directory + '/SWaT_Attack_pid1.csv',
                 directory + '/SWaT_Attack_pid2.csv',
                 directory + '/SWaT_Attack_pid3.csv',
                 directory + '/SWaT_Attack_pid4.csv',
                 directory + '/SWaT_Attack_pid5.csv']
    predict_file = ''
    extend_head = 0
    extend_tail = 0
    correct_ratio_d = 0.0
    correct_ratio_f = 0.5
    multi_hit = True
    preceding_process = 0

    paramlist = []
    paramfile = open('parameters.cfg', 'r', encoding='utf-8', newline='')
    for line in paramfile.readlines():
        tokens = line.strip().split('=')
        paramlist.append(tokens[1])
    paramfile.close()

    #attack_file = paramlist[0]
    extend_head = int(paramlist[0])
    extend_tail = int(paramlist[1])
    correct_ratio_d = float(paramlist[2])
    correct_ratio_f = float(paramlist[3])
    if paramlist[4] == 'True':
        multi_hit = True
    elif paramlist[4] == 'False':
        multi_hit = False
    preceding_process = int(paramlist[5])

    try:
        opts, args = getopt.getopt(argv, "hi:a:", ["input file=", "attack file="])
    except getopt.GetoptError:
        print('Error')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('evaluation.py -i <input_file> -a <attack_file>')
            sys.exit()
        elif opt in ("-i"):
            predict_file = arg
        elif opt in ("-a"):
            attack_file = arg

    if len(predict_file) == 0:
        print('Error: Input the result file with option -i')
        return

    if len(attack_file) == 0:
        print('Error: Input the attack file (ground truth) with option -a')
        return

    head_str = [ 'Overall Result',
                 'Result for Process1',
                 'Result for Process2',
                 'Result for Process3',
                 'Result for Process4',
                 'Result for Process5',
                 'Result for Process6' ]

    attack_lists = []
    for idx in range(len(attack_file)):
        attack_lists.append(read_period_file(attack_file[idx]))
    predict_lists = read_timeseries_file(predict_file)

    eval = Evaluation()
    eval.set_prediction(predict_lists[0])
    eval.set_ground_truth(attack_lists[0])
    eval.extend_ground_truth(extend_head=extend_head, extend_tail=extend_tail)

    print(head_str[0])
    #print('#Predictions: ' + str(eval.get_n_predictions()))
    eval.detected_attacks(correct_ratio_d, multi_hit)
    eval.false_alarm(correct_ratio_f, multi_hit)
    print('', flush=True)

    if len(predict_lists) > 1:
        eval = Evaluation_SWaT_Process()
        eval.set_prediction(predict_lists[1:])
        eval.set_ground_truth(attack_lists[1:])
        eval.extend_ground_truth(extend_head=extend_head, extend_tail=extend_tail)
    
        for idx in range(len(predict_lists)-1):
            print(head_str[idx+1])
            #print('#Predictions: ' + str(eval.get_n_predictions(idx)))
            eval.detected_attacks(idx, preceding_process, correct_ratio_d, multi_hit)
            eval.false_alarm(idx, preceding_process, correct_ratio_f, multi_hit)
    
            print('', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])

