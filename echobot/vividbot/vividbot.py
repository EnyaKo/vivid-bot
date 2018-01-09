import os
import itertools
import logging
from typing import List, Dict

from nltk import FreqDist
from nltk.util import ngrams
import jieba
from gensim import models

from .big5trait import big5trait

from ..models import BiGramRelation, Phrase


logger = logging.getLogger(__name__)


class VividBot:
    def __init__(self, *, static_root, word2vec_dic_path, jieba_dic_path, stopword_path):
        self.STATIC_BASE = os.path.join(static_root, 'echobot')

        dict_path = os.path.join(self.STATIC_BASE, jieba_dic_path)
        file_path = os.path.join(self.STATIC_BASE, word2vec_dic_path)
        stopword_path = os.path.join(self.STATIC_BASE, stopword_path)
        self._jieba_init(dict_path)
        self._load_stopwords(stopword_path)
        self._load_word2vec(file_path)

    def _jieba_init(self, dict_path):
        jieba.initialize()
        jieba.set_dictionary(dict_path)

    def _load_stopwords(self, stopword_path):
        with open(stopword_path) as stopword_file:
            self.stopwords = {line for line in stopword_file.readlines()}

    def _load_word2vec(self, word2vec_dic_path):
        with open(word2vec_dic_path, 'rb') as word2vec_file:
            self.model = models.Word2Vec.load_word2vec_format(word2vec_file, binary=True)

    def feed(self, text):
        output = self._split_word(text)
        logger.info('jieba ouptut: {}'.format(output))
        thes_dic = self._thesaurus(output)
        logger.info('word2vec output: {}'.format(thes_dic))
        filtered_dic = big5trait(thes_dic, [0.10, 0.10, -0.02, 0.04, 0.06], self.STATIC_BASE)
        logger.info('big5 output: {}'.format(filtered_dic))
        text = self._generate_reponse_from_bi_gram(output, filtered_dic)
        return text

    def addbig5(req):
        pass

    def _split_word(self, s, mode=0):
        result = []
        seg_list = jieba.cut(s)
        if mode is not 0:
            for seg in seg_list:
                if seg not in self.stopwords:
                    result.append(seg)
        else:
            for seg in seg_list:
                result.append(seg)
        return result

    def _thesaurus(self, output):
        thes_dic = dict()
        for q_items in output:
            thes_list = []
            try:
                thes = self.model.most_similar(q_items, topn=100)
                thes_list.append(q_items)
                for items in thes:
                    thes_list.append(items[0])
            except Exception:
                    thes_list.append(q_items)
            thes_dic[q_items] = thes_list
        return thes_dic

    @staticmethod
    def _train_bi_gram(split_sentence: List[str]):
        bi_freq = FreqDist(ngrams(split_sentence, 2))

        total_num = len(bi_freq)
        for index, ((pre, post), freq) in enumerate(bi_freq.items()):
            pre_phrase = Phrase.objects.get_or_create(content=pre)[0]
            post_phrase = Phrase.objects.get_or_create(content=post)[0]
            try:
                phrase_relation = BiGramRelation.objects.get(prev=pre_phrase,
                                                             post=post_phrase)
                phrase_relation.freq += freq
            except BiGramRelation.DoesNotExist:
                phrase_relation = BiGramRelation(
                    prev=pre_phrase,
                    post=post_phrase,
                    freq=freq
                )
                phrase_relation.save()
            finally:
                logger.info('Write BiGram ({}/{}): {} - {} ({})'.format(
                    index, total_num, pre, post, freq
                ))

    @staticmethod
    def _generate_reponse_from_bi_gram(split_sentence: List[str],
                                       sim_words_dict: Dict[str, List[str]],
                                       thresh: int=3) -> str:
        """
        Generate Response Using BiGram

        Args:
            split_sentence: Split sentence from jieba
            sim_word_dict: Similiar Words Dict
            thresh: The threshold of simliar words.
                    The words' rank lower than thershold won't be used.

        Returns:
            Best response from bigram
        """

        # Set origin response as default response
        best_response = [sim_words_dict[word][0] for word in split_sentence]
        best_freq = 1
        freq_cache = dict()

        group_of_words = [sim_words_dict[word][:thresh] for word in split_sentence]
        for combination in itertools.product(*group_of_words):
            freq = 1

            for pre_word, post_word in zip(combination[:-1], combination[1:]):
                cur_freq = freq_cache.get((pre_word, post_word), None)
                if cur_freq:
                    freq *= cur_freq
                    continue

                try:
                    pre_phrase = Phrase.objects.get(content=pre_word)
                    post_phrase = Phrase.objects.get(content=post_word)
                except Phrase.DoesNotExist:
                    logger.info('{} or {} does not exist'.format(pre_word, post_word))
                    # TODO: Add punishment for pattern not exist
                    pass
                else:
                    try:
                        bi_gram_relation = BiGramRelation.objects.get(
                            prev=pre_phrase,
                            post=post_phrase
                        )
                    except BiGramRelation.DoesNotExist:
                        # TODO: Add punishment for pattern not exist
                        pass
                    else:
                        freq *= bi_gram_relation.freq
                        freq_cache[(pre_word, post_word)] = freq

            if freq > best_freq:
                best_freq = freq
                best_response = combination 
        return ''.join(best_response)
