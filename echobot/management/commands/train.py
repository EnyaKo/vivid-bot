from django.core.management.base import BaseCommand
from django.conf import settings

import logging
import os

from gensim.models import word2vec
import jieba

from echobot.vividbot import VividBot


class Command(BaseCommand):
    help = 'Django admin custom command poc.'

    def add_arguments(self, parser):
        parser.add_argument(
            'path'
        )

    def handle(self, *args, **options):
        static_root = settings.STATIC_ROOT
        STATIC_BASE = os.path.join(static_root, 'echobot')
        #train_path = 'train/ptt_corpus.txt'

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        # jieba initialize settings
        jieba_dict_path = os.path.join(STATIC_BASE, 'data/dict.txt.big')
        jieba.set_dictionary(jieba_dict_path)
        stopwordset = set()
        stopword_path = os.path.join(STATIC_BASE, 'data/stop_words.txt')
        with open(stopword_path, 'r', encoding='utf-8') as sw:
            for line in sw:
                stopwordset.add(line.strip('\n'))

        # text to segments
        texts_num = 0
        split_sentence = []
        #text_file_path = os.path.join(STATIC_BASE, options['path'])
        with open(options['path'],'r') as content :
            for line in content:
                line = line.strip('\n')
                words = jieba.cut(line, cut_all=False)
                for word in words:
                    if word not in stopwordset:
                        split_sentence.append(word)

                texts_num += 1
                if texts_num % 10000 == 0:
                    logging.info("finished cutting %d lines" % texts_num)

        # train word2vec model
        model = word2vec.Word2Vec(split_sentence, size=250)

        # save word2vec model(can be reused)
        word2vec_model_path = os.path.join(STATIC_BASE, 'med250_ptt.model.bin')
        model.save_word2vec_format(word2vec_model_path, binary=True)

        # train bi_gram
        VividBot._train_bi_gram(split_sentence)
