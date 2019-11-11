#!/home/pzhang/anaconda3/bin/python3.5

import os
from torch.utils.data import Dataset
from multiprocessing import Process, Queue, Event
from dataset.kaldi_io import FeatureReader
import time
import numpy as np
import random


class DataOutOfRange(Exception):
    pass


def get_speaker_info(data, spklist):
    """Get speaker information from the data directory.

    This function will be used in KaldiDataReader and KaldiDataQueue. So make it a normal function rather than a class
    method would be fine.

    Args:
        data: The kaldi data directory.
        spklist: The spklist file gives the index of each speaker.
    :return:
        spk2features: A dict. The key is the speaker id and the value is the segments belonging to this speaker.
        features2spk: A dict. The key is the segment and the value is the corresponding speaker id.
        spk2index: A dict from speaker NAME to speaker ID. This is useful to get the number of speakers. Because
                   sometimes, the speakers are not all included in the data directory (like in the valid set).

        segment format: "utt_name filename:offset"
    """
    assert (os.path.isdir(data) and os.path.isfile(spklist))
    spk2index = {}
    with open(spklist, "r") as f:
        for line in f.readlines():
            spk, index = line.strip().split(" ")
            spk2index[spk] = int(index)

    utt2spk = {}
    with open(os.path.join(data, "spk2utt"), "r") as f:
        for line in f.readlines():
            spk, utts = line.strip().split(" ", 1)
            for utt in utts.split(" "):
                utt2spk[utt] = spk2index[spk]

    spk2features = {}
    features2spk = {}
    with open(os.path.join(data, "feats.scp"), "r") as f:
        for line in f.readlines():
            line = bytes(line, encoding='utf-8')
            (key, rxfile) = line.decode().split(' ')  # 有问题
            spk = utt2spk[key]
            if spk not in spk2features:
                spk2features[spk] = []
            spk2features[spk].append(key + ' ' + rxfile)
            features2spk[key + ' ' + rxfile] = spk
    return spk2features, features2spk, spk2index


def batch_random(stop_event,
                 queue,
                 data,
                 spk2features,
                 num_total_speakers,
                 num_speakers=10,
                 num_segments=10,
                 min_len=200,
                 max_len=400,
                 shuffle=True,
                 seed=0):
    """Load features and fill a queue. Used in KaldiDataRandomQueue

    Args:
        stop_event: An event to tell the process to stop.
        queue: A queue to put the data.
        data: The kaldi data directory.
        spk2features: A dict from speaker index to the segments.
        num_total_speakers: The total number of speakers.
        num_speakers: The number of speakers in the batch.
        num_segments: The number of segments per speaker.
        min_len: The minimum length of the features.
        max_len: The maximum length of the features.
        shuffle: Load the feature from the 0-th frame or a random frame.
        seed: The value used to generate the random seed.
    """
    # TODO: If you use numpy.random in the sub-process, it is better to use:
    # local_state = np.random.RandomState(seed)
    # print local_state.uniform(0, 1, 5)
    #
    # The re-seed is necessary if numpy.random is used
    # You can use os.urandom to generate the `random` seed.
    rd = random.Random(os.urandom(4))
    rd.seed(seed)

    feature_reader = FeatureReader(data)
    speakers = list(spk2features.keys()) # 7323
    if num_total_speakers < num_speakers:
        print(
            "[Warning] The number of available speakers are less than the required speaker. Some speakers will be duplicated.")
        speakers = speakers * (int(num_speakers / num_total_speakers) + 1)
    # Now we have enough speakers
    while not stop_event.is_set():
        batch_speakers = rd.sample(speakers, num_speakers) # 为选出的spk_id
        batch_length = rd.randint(min_len, max_len)  # 在min_len 200 和max_len 400之间随机选择一个batch_length
        features = np.zeros((num_speakers * num_segments, batch_length, feature_reader.dim), dtype=np.float32)  # (batch_size, frame_length, feat_dim)
        labels = np.zeros((num_speakers * num_segments), dtype=np.int32) # (batch_size)
        for i, speaker in enumerate(batch_speakers):
            # The length may be larger than the utterance length. A check should be applied first.
            feature_list = []
            spk = speaker
            while len(feature_list) == 0:
                feature_list = []
                for feat in spk2features[spk]:
                    if feature_reader.utt2num_frames[feat.split(' ')[0]] > batch_length:
                        feature_list.append(feat)
                if len(feature_list) == 0:
                    # The speaker is not appropriate for this batch. Resample the speaker
                    spk = rd.choice(list(set(speakers) - set(batch_speakers)))
                    batch_speakers[i] = spk

            labels[i * num_segments:(i + 1) * num_segments] = spk
            # If the number is not enough
            if len(feature_list) < num_segments:
                feature_list *= (int(num_segments / len(feature_list)) + 1)  # 对现有的list进行复制
            # Now the length of the list must be greater than the sample size.
            speaker_features = rd.sample(feature_list, num_segments)  # 从现有该说话人的feature_list中选出num_segments句作为speaker features
            for j, feat in enumerate(speaker_features):
                features[i * num_segments + j, :, :], _ = feature_reader.read_segment(feat, batch_length, shuffle=shuffle)
        queue.put((features, labels))

    time.sleep(3)
    while not queue.empty():
        try:
            queue.get(block=False)
        except:
            pass
    print("The process {} is about to exit.".format(os.getpid()))
    return


class KaldiDataRandomQueue(object):
    """A queue to read features from Kaldi data directory."""

    def __init__(self, data_dir, spklist, num_parallel=1, max_qsize=10, num_speakers=None, num_segments=None, min_len=None, max_len=None, shuffle=True):
        """ Create a queue from a given directory.

        This is basically similar with KaldiDataRead. The difference is that KaldiDataReader uses tf.data to load
        features and KaldiDataQueue uses multiprocessing to load features which seems to be a better choice since
        the multiprocessing significantly speed up the loading in my case. If you can make parallel_interleave works,
        it is definitely more convenient to use KaldiDataReader because it's more simple.

        Args:
            data_dir: The kaldi data directory.
            spklist: The spklist tells the mapping from the speaker name to the speaker id.
            num_parallel: The number of threads to read features.
            max_qsize: The capacity of the queue
            num_speakers: The number of speakers per batch.
            num_segments: The number of semgents per speaker.
              batch_size = num_speakers * num_segments
              When num_segments = 1, the batch is randomly chosen from n speakers,
              which is used for softmax-like loss function. While we can sample multiple segments for each speaker,
              which is used for triplet-loss or GE2E loss.
            min_len: The minimum length of the sampled sequence.
            max_len: The maximum length of the sampled sequence.
            shuffle: Loading data from the 0-th frame or a random frame.
        """
        self.data = data_dir
        self.num_speakers = num_speakers
        self.num_segments = num_segments
        self.min_len = min_len
        self.max_len = max_len
        self.num_parallel_datasets = num_parallel
        self.shuffle = shuffle

        # We process the data directory and fetch speaker information.
        self.spk2features, self.features2spk, spk2index = get_speaker_info(data_dir, spklist)

        # The number of speakers should be
        self.num_total_speakers = len(list(spk2index.keys()))

        # The Queue is thread-safe and used to save the features.
        self.queue = Queue(max_qsize)
        self.stop_event = Event()

        # And the prcesses are saved
        self.processes = []

    def set_batch(self, num_speakers, num_segments):
        """Set the batch-related parameters

        Args:
            num_speakers: The number of speakers per batch.
            num_segments: The number of semgents per speaker.
        """
        self.num_speakers = num_speakers
        self.num_segments = num_segments

    def set_length(self, min_len, max_len):
        """Set the length of the sequence

        Args:
            min_len: The minimum length of the sampled sequence.
            max_len: The maximum length of the sampled sequence.
        """
        self.min_len = min_len
        self.max_len = max_len

    def start(self):
        """Start processes to load features
        """
        self.processes = [Process(target=batch_random, args=(self.stop_event,
                                                             self.queue,
                                                             self.data,
                                                             self.spk2features,
                                                             self.num_total_speakers,
                                                             self.num_speakers,
                                                             self.num_segments,
                                                             self.min_len,
                                                             self.max_len,
                                                             self.shuffle,
                                                             i))
                          for i in range(self.num_parallel_datasets)]
        for process in self.processes:
            process.daemon = True
            process.start()

    def fetch(self):
        """Fetch data from the queue，从队列中取数据"""
        return self.queue.get()

    def stop(self):
        """Stop the threads

        After stop, the processes are terminated and the queue may become unavailable.
        """
        self.stop_event.set()
        print("Clean the data queue that subprocesses can detect the stop event...")
        while not self.queue.empty():
            # Clear the queue content before join the threads. They may wait for putting the data to the queue.
            self.queue.get()
        time.sleep(3)
        for process in self.processes:
            # TODO: fix the join problem
            process.terminate()
            # process.join()


class KaldiDataSeqQueue(object):
    """A queue to read features from Kaldi data directory."""

    def __init__(self, data_dir, spklist, num_parallel=1, max_qsize=10, batch_size=128, min_len=None, max_len=None, shuffle=True):
        """ Create a queue from a given directory.

        Unlike KaldiDataRandomQueue, KaldiDataSeqQueue load data in sequence which means each segment appears once
        in one epoch. This is usually used for validation (using softmax-like loss or EER).

        Args:
            data_dir: The kaldi data directory.
            spklist: The spklist tells the mapping from the speaker name to the speaker id.
            num_parallel: The number of threads to read features.
            max_qsize: The capacity of the queue.
            batch_size: The batch size.
            min_len: The minimum length of the sampled sequence.
            max_len: The maximum length of the sampled sequence.
            shuffle: Shuffle the load sequence and loading data from a random frame.
        """
        self.data = data_dir
        self.batch_size = batch_size
        self.min_len = min_len
        self.max_len = max_len
        self.num_parallel_datasets = num_parallel
        self.shuffle = shuffle

        # We process the data directory and fetch speaker information.
        self.spk2features, self.features2spk, spk2index = get_speaker_info(data_dir, spklist)
        self.num_total_speakers = len(list(spk2index.keys()))

        # Arrange features in sequence
        self.feature_list = []
        self.sub_feature_list = []
        for spk in self.spk2features:
            self.feature_list += self.spk2features[spk]

        if shuffle:
            random.shuffle(self.feature_list)
        # Split the features to N sub-list. The lists are used in each process.
        num_sub_features = int(len(self.feature_list) / num_parallel)  # 这块要求是个整数
        for i in range(num_parallel):
            if i == num_parallel - 1:
                self.sub_feature_list.append(self.feature_list[i * num_sub_features:])
            else:
                self.sub_feature_list.append(self.feature_list[i * num_sub_features:(i + 1) * num_sub_features])

        # The Queue is thread-safe and used to save the features.
        self.queue = Queue(max_qsize)

        # The events will be set once the processes finish its job
        self.stop_event = [Event() for _ in range(num_parallel)]

        # And the prcesses are saved
        self.processes = []

    def set_batch(self, batch_size):
        """Set the batch size
        """
        self.batch_size = batch_size

    def set_length(self, min_len, max_len):
        """Set the length of the sequence

        Args:
            min_len: The minimum length of the sampled sequence.
            max_len: The maximum length of the sampled sequence.
        """
        self.min_len = min_len
        self.max_len = max_len

    def start(self):
        """Start processes to load features
        """
        self.processes = [Process(target=batch_sequence, args=(self.stop_event[i],
                                                               self.queue,
                                                               self.data,
                                                               self.sub_feature_list[i],
                                                               self.features2spk,
                                                               self.batch_size,
                                                               self.min_len,
                                                               self.max_len,
                                                               self.shuffle,
                                                               i))
                          for i in range(self.num_parallel_datasets)]
        for process in self.processes:
            process.daemon = True
            process.start()

    def fetch(self):
        """Fetch data from the queue"""
        if self.queue.empty():
            all_finish = [self.stop_event[i].is_set() for i in range(self.num_parallel_datasets)]
            if all(all_finish):
                # If the queue is empty and all processes are finished, we got nothing to read.
                for process in self.processes:
                    # TODO: fix the join problem
                    process.terminate()
                raise DataOutOfRange

        return self.queue.get()

    def stop(self):
        """Stop the threads"""
        for process in self.processes:
            # TODO: fix the join problem
            process.terminate()
            # process.join()


def batch_sequence(stop_event,
                   queue,
                   data,
                   feature_list,
                   features2spk,
                   batch_size=128,
                   min_len=200,
                   max_len=400,
                   shuffle=True,
                   seed=0):
    """Load features and fill a queue. Used in KaldiDataSeqQueue.

    Args:
        stop_event: An event indicating the reading is finished.
        queue: A queue to put the data.
        data: The kaldi data directory.
        feature_list: A list shows which features the process should read.
        features2spk: A dict map features to speaker index.
        batch_size: The batch_size
        min_len: The minimum length of the features.
        max_len: The maximum length of the features.
        shuffle: Load the feature from the 0-th frame or a random frame.
        seed: The number is used to generate a random seed
    """
    # Read the comment in batch_random
    rd = random.Random(os.urandom(4))
    rd.seed(seed)

    # rd.jumpahead(seed)

    feature_reader = FeatureReader(data)
    num_batches = int(len(feature_list) / batch_size)
    for i in range(num_batches):
        batch_length = rd.randint(min_len, max_len)

        # In some cases, the minimum length of the utterances is smaller than the batch length.
        # Use the smallest length as the real batch length.
        for j in range(batch_size):
            if feature_reader.utt2num_frames[feature_list[i * batch_size + j].split(' ')[0]] < batch_length:
                batch_length = feature_reader.utt2num_frames[feature_list[i * batch_size + j].split(' ')[0]]

        features = np.zeros((batch_size, batch_length, feature_reader.dim), dtype=np.float32)
        labels = np.zeros((batch_size), dtype=np.int32)
        for j in range(batch_size):
            features[j, :, :], _ = feature_reader.read_segment(feature_list[i * batch_size + j], batch_length, shuffle=shuffle)
            labels[j] = features2spk[feature_list[i * batch_size + j]]
        queue.put((features, labels))
    stop_event.set()
    print("The process {} is about to exit.".format(os.getpid()))
    return

