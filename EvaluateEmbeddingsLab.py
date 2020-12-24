import csv
import numpy as np

from scipy.spatial import distance
from scipy import stats


class Wordpair:
    word1 = None
    word2 = None
    
    def __init__(self, word1, word2):
        self.word1 = word1
        self.word2 = word2
    
    def __hash__(self):
        return hash(self.word1 + ',' + self.word2)
    
    def __eq__(self, other):
        return isinstance(other, type(self)) and hash(self) == hash(other)
    
    def inverse(self):
        return Wordpair(self.word2, self.word1)
    
    def to_list_of_strings(self):
        return [self.word1, self.word2]
    
    def to_string(self, separator=','):
        return self.word1 + separator + self.word2
    
    def __str__(self):
        return self.to_string()
    
    def __getitem__(self, index):
        if index == 0:
            return self.word1
        elif index == 1:
            return self.word2
        else:
            raise IndexError
    
    def word_in_pair(self, word):
        return self.word1 == word or self.word2 == word

    
class Labelpair(Wordpair):
    """
    Labels are expected to be POS (part of speech)
    
    """
    def __init__(self, label1, label2):
        super().__init__(label1, label2)
        
        
class Data:
    """
    Vault for all word similarities with labels
    
    """
    data_similarity = None
    data_labels = None
    
    def __init__(self, similarities=None, labels=None):
        if similarities is None:
            self.data_similarity = {}
        else:
            self.data_similarity = similarities
        
        if labels is None:
            self.data_labels = {}
        else:
            self.data_labels = labels
            
    def __len__(self):
        return len(self.data_similarity)
        
    def add(self, wordpair, labelpair, value):
        if type(wordpair) == tuple and type(labelpair) == tuple:
            return self.add(Wordpair(wordpair[0], wordpair[1]), Wordpair(labelpair[0], labelpair[1]), value)
        
        self.data_similarity[wordpair] = value
        self.data_labels[wordpair] = labelpair
        
    def pop(self, wordpair):
        if type(wordpair) == tuple or type(wordpair) == list:
            return self.pop(Wordpair(wordpair[0], wordpair[1]))
        
        self.data_similarity.pop(wordpair)
        self.data_labels.pop(wordpair)
        
    def get(self, wordpair):
        if type(wordpair) == tuple or type(wordpair) == list:
            return self.get(Wordpair(wordpair[0], wordpair[1]))
        
        return self.data_similarity[wordpair]
        
    def get_with_labels(self, wordpair):
        if type(wordpair) == tuple or type(wordpair) == list:
            return self.get_with_labels(Wordpair(wordpair[0], wordpair[1]))
        
        return [self.data_labels[wordpair].to_list_of_strings(), self.data_similarity[wordpair]]
    
    def to_list(self):
        result = []
        all_pairs = self.data_similarity.keys()
        
        for pair in all_pairs:
            curr_list = pair.to_list_of_strings() + \
                self.data_labels[pair].to_list_of_strings() + \
                [float(self.data_similarity[pair])]
            result.append(curr_list)
            
        return result
    
    def get_all_pairs_with_word(self, word):
        res = []
        
        for pair in self.data.data_similarity:
            if pair.word_in_pair(word):
                res.append(pair)
        
        return res

    
class Dataset:
    """ 
    Base class for dataset
    
    """
    path = None
    
    def __init__(self, path):
        self.path = path
        
    def load_data_to_memory(self):
        raise NotImplementedError

        
class GoldenStandartDataset(Dataset):
    """
    Desribes arbitrary golden standart
    
    """
    standartized_label = "standartized"
    data = None
 
    def __init__(self, path, data=None):
        super().__init__(path)
        
        if data is None:
            self.data = Data()
            self.load_data_to_memory()
        else:
            self.data = data
            
    def load_data_to_memory(self):
        if self.standartized_label in self.path:
            self.load_data_to_memory_standartized()
        else:
            raise NotImplementedError
            
    def load_data_to_memory_standartized(self):
        """
        Read data from path in standartized form:
        word1, word2, label1, label2, similarity value
        
        labels are expected to be POS (part of speech)
        
        """
        separator = ','
        
        with open(self.path, newline='\n') as csv_file:
            reader = csv.reader(csv_file, delimiter=separator)
            for row in reader:
                words = Wordpair(row[0], row[1])
                labels = Labelpair(row[2], row[3])
                sim_value = row[4]
                
                try:
                    float(sim_value)
                except ValueError:
                    continue
                
                self.data.add(words, labels, sim_value)
    
    def write_data_to_file(self, filepath):
        """
        Write data to filepath in standartized form:
        word1, word2, label1, label2, similarity value

        labels are expected to be POS (part of speech)

        """
        separator = ','
        
        with open(filepath, 'w+', newline='\n') as csv_file:
            writer = csv.writer(csv_file, delimiter=separator)
            writer.writerow(["word1", "word2", "label1", "label2", "sim_value"])
            writer.writerows(self.data.to_list())
    
    def find_asymm_pairs(self, printing=True):
        """
        Find asymm pairs like (w1, w2, v1) & (w2, w1, v2), where v1 != v2
        
        """
        arr = [key for key in self.data.data_similarity.keys()]
        reversed_arr = [key.inverse() for key in self.data.data_similarity.keys()]
        
        direct_set = set(arr)
        reversed_set = set(reversed_arr)
        
        intersected_keys = list(set.intersection(direct_set, reversed_set))

        sym_keys = []

        for key in intersected_keys:
            if key.inverse() in sym_keys or key.inverse() == key:
                continue
            sym_keys.append(key)

        if len(sym_keys) == 0 and printing:
            print('no asymmetrical pairs')

        for key in sym_keys:
            val1 = self.data.data_similarity[key]
            val2 = self.data.data_similarity[key.inverse()]
            if printing:
                print(str(key) + ',' + str(val1), ';', str(key.inverse()) + ',' + str(val2))

        return sym_keys
    
    def find_all_symm_keys(self):
        """
        Find symm pairs like (w1, w2) & (w2, w1)
        
        """
        arr = [key for key in self.data.data_similarity.keys()]
        reversed_arr = [key.inverse() for key in self.data.data_similarity.keys()]
        
        direct_set = set(arr)
        reversed_set = set(reversed_arr)
        
        intersected_keys = list(set.intersection(direct_set, reversed_set))
        
        return intersected_keys
    
    def calculate_POS_distribution(self):
        """
        Calculate words part of speech distribution
        
        """
        speech_parts = {}
        
        for label_pair in self.data.data_labels.values():
            for label in label_pair.to_list_of_strings():
                if label not in speech_parts:
                    speech_parts[label] = 1
                else:
                    speech_parts[label] += 1

        for part in speech_parts:
            speech_parts[part] /= (len(self.data.data_labels) * 2)

        return speech_parts
    
    def get_all_words_and_labels(self):
        """
        Get list of all unique words & list of their labels
        
        """
        all_words = []
        all_words_labels = []
        
        for pair in self.data.data_labels.keys():
            label_pair = self.data.data_labels[pair]
            
            for word_idx in [0, 1]:
                if pair[word_idx] not in all_words:
                    all_words.append(pair[word_idx])
                    all_words_labels.append(label_pair[word_idx])
            
        return all_words, all_words_labels
    
    
    def train_test_split(self, test_ratio, only_test_words_ratio=0.2):
        """
        Split all pairs to train & test with respect to original POS distribution
        
        Parameters:
        test_ratio : part of pair that goes to test
        only_test_words_ratio : part of words that exists only in test
        
        Example:
        
        imagine original dataset with 1000 pairs & 800 unique words
        test_ratio=0.3 means that test should contain 300 pairs
        only_test_words_ratio=0.1 means that test should contain 80 words existing only in test
        
        """
        POS_distr = self.calculate_POS_distribution()
        
        all_words, all_words_labels = self.get_all_words_and_labels()
        
        all_words_size = len(all_words)
        only_test_words_size = int(all_words_size * only_test_words_ratio)
        test_size = int(len(self.data.data_similarity) * test_ratio)
        
        test_dict = Data()
        
        for idx in range(only_test_words_size):
            if (len(test_dict) >= test_size):
                break
                
            curr_label = np.random.choice(list(POS_distr.keys()), p=list(POS_distr.values()))
            possible_words_idxs = np.where(np.array(all_words_labels) == curr_label)[0]
            curr_word_idx = np.random.choice(possible_words_idxs)
            curr_word = all_words[curr_word_idx]
            all_words.pop(curr_word_idx)
            all_words_labels.pop(curr_word_idx)
            
            for pair in self.data.data_similarity:
                if pair.word_in_pair(curr_word):
                    test_dict.add(pair, self.data.data_labels[pair], self.data.data_similarity[pair])
            
        gap_size = test_size - len(test_dict)
                
        if gap_size > 0:

            while len(test_dict) < test_size:
                curr_label = np.random.choice(list(POS_distr.keys()), p=list(POS_distr.values()))
                possible_words_idxs = np.where(np.array(all_words_labels) == curr_label)[0]
                curr_word_idx = np.random.choice(possible_words_idxs)
                curr_word = all_words[curr_word_idx]

                potencial_pairs = []
                for pair in self.data.data_similarity:
                    if pair.word_in_pair(curr_word):
                        potencial_pairs.append(pair)
                
                if len(potenial_pairs <= 1):
                    continue

                pair = np.random.choice(potencial_pairs)
                
                if len(self.get_all_pairs_with_word(self, pair[0])) <= 1 or \
                len(self.get_all_pairs_with_word(self, pair[1])) <= 1:
                    continue

                if pair not in test_dict.data_similarity and len(get_all_pairs_with_word(curr_word)):
                    test_dict.add(pair, self.data.data_labels[pair], self.data.data_similarity[pair])
        
        train_dict = Data()
        for pair in self.data.data_similarity:
            if pair not in test_dict.data_similarity:
                train_dict.add(pair, self.data.data_labels[pair], self.data.data_similarity[pair])
        
        return GoldenStandartDataset(self.path + "train", train_dict), GoldenStandartDataset(self.path + "test", test_dict)
    
    
class SimLex999Dataset(GoldenStandartDataset):

    def __init__(self, path="./SimLex-999.txt"):
        super().__init__(path)
    
    def load_data_to_memory(self):
        if self.path.endswith(self.standartized_label):
            super().load_data_to_memory()
        else:
            separator = '\t'
            
            with open(self.path, newline='\n') as csv_file:
                reader = csv.reader(csv_file, delimiter=separator)
                
                line_idx = 0
                for row in reader:
                    if (line_idx == 0):
                        line_idx += 1
                        continue
                    
                    words = Wordpair(row[0], row[1])
                    labels = Labelpair(row[2], row[2])
                    sim_value = float(row[3])

                    self.data.add(words, labels, sim_value)
                    
                    
class WordSim353Dataset(GoldenStandartDataset):

    def __init__(self, path="./combined.csv"):
        super().__init__(path)
    
    def load_data_to_memory(self):
        if self.path.endswith(self.standartized_label):
            super().load_data_to_memory()
        else:
            separator = ','
            
            with open(self.path, newline='\n') as csv_file:
                reader = csv.reader(csv_file, delimiter=separator)
                
                line_idx = 0
                for row in reader:
                    if (line_idx == 0):
                        line_idx += 1
                        continue
                    
                    words = Wordpair(row[0], row[1])
                    labels = Labelpair("n", "n")
                    sim_value = float(row[2])

                    self.data.add(words, labels, sim_value)

                    
class MENDataset(GoldenStandartDataset):
    def __init__(self, path="./MEN_dataset_lemma_form_full"):
        super().__init__(path)
    
    def load_data_to_memory(self):
        if self.path.endswith(self.standartized_label):
            super().load_data_to_memory()
        else:
            separator = ' '
            
            with open(self.path, newline='\n') as csv_file:
                reader = csv.reader(csv_file, delimiter=separator)

                for row in reader:
                    
                    words = Wordpair(row[0][:-2], row[1][:-2])
                    labels = Labelpair(row[0][-1], row[1][-1])
                    sim_value = float(row[2])

                    self.data.add(words, labels, sim_value)
                    
                    
class EmbeddingEvaluator:
    golden_data = None
    golden_data_name = None
    fit_data = None
    dist_func = None
    
    def __init__(self, golden_path=None, golden_standart=None):
        if golden_path is None and golden_standart is None:
            raise NotImplementedError
        
        if golden_standart is not None:
            golden_data_name = golden_standart
            if golden_standart == "SimLex-999":
                self.golden_data = SimLex999Dataset()
            elif golden_standart == "WordSim-353":
                self.golden_data = WordSim353Dataset()
            elif golden_standart == "MEN":
                self.golden_data = MENDataset()
            else:
                raise NotImplementedError
        
        if golden_path is not None:
            self.golden_data = GoldenStandartDataset(golden_path)
        self.fit_data = Data()
            
    def fit(self, embeddings, dist_func=None):
        """
        Saves embeddings to evaluate
        
        Parameters:
        embeddings should support word indexing, like that:
        embeddings['word'] = word embedding
        
        """
        if dist_func is None:
            self.dist_func = distance.cosine
        else:
            self.dist_func = dist_func
        self.fit_data = embeddings
        
    def evaluate(self, method='spearman'):
        """
        Evaluate embeddings quality
        
        Parameters:
        method='spearman' : calculate rank correlation between golden standart & embeddings
        
        """
        if self.fit_data is None:
            raise NotImplementedError
        if method == 'spearman':
            golden_scores = []
            fitted_scores = []
            for pair in self.golden_data.data.data_similarity:
                golden_scores.append(self.golden_data.data.data_similarity[pair])
                emb1 = self.fit_data[pair[0]]
                emb2 = self.fit_data[pair[1]]
                
                if type(emb1) == map:
                    emb1 = list(emb1)
                if type(emb2) == map:
                    emb2 = list(emb2)
                if len(emb1) == 0 or len(emb2) == 0:
                    # no such word in given embeddings
                    raise NotImplementedError

                fitted_scores.append(self.dist_func(np.array(emb1), np.array(emb2)))
                
            corr = stats.spearmanr(np.array(golden_scores), np.array(fitted_scores))
            print("Spearman correlation on {}: {}".format(self.golden_data_name, corr))
            
        else:
            raise NotImplementedError