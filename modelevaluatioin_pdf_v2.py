# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 01:47:20 2020

@author: ECE590windows
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import ntpath
import PyPDF2
from tmtoolkit.corpus import Corpus
from tmtoolkit.preprocess import TMPreproc
import numpy as np
import logging
import warnings
from tmtoolkit.topicmod.tm_lda import compute_models_parallel

pdf_dict = {}
for white_paper_pdf in glob.glob('/Users/SaiBhargavi/Desktop/PDF_Python/white papers_2/*.pdf'):
    key = ntpath.basename(white_paper_pdf).split('.')[0]
    pdfFileObj = open(white_paper_pdf, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    text = ''
    for k in range(pdfReader.numPages):
        text = text + ' ' + pdfReader.getPage(k).extractText()
    if text.strip() != '':
        pdf_dict[key]=text

corpus = Corpus(pdf_dict)

preproc = TMPreproc(corpus, language='en')
preproc.pos_tag() \
    .lemmatize() \
    .tokens_to_lowercase() \
    .remove_special_chars_in_tokens()

preproc_bigger = preproc.copy() \
    .add_stopwords(['would', 'could', 'nt', 'mr', 'mrs', 'also', 'token']) \
    .clean_tokens(remove_shorter_than=2) \
    .remove_common_tokens(df_threshold=0.85) \
    .remove_uncommon_tokens(df_threshold=0.05)

preproc_bigger.n_docs, preproc_bigger.vocabulary_size

doc_labels = np.array(preproc_bigger.doc_labels)
#doc_labels[:10]
vocab_bg = np.array(preproc_bigger.vocabulary)
dtm_bg = preproc_bigger.dtm
#preproc_bigger.shutdown_workers()
#del preproc_bigger
#dtm_bg

# suppress the "INFO" messages and warnings from lda
logger = logging.getLogger('lda')
logger.addHandler(logging.NullHandler())
logger.propagate = False

warnings.filterwarnings('ignore')

# and fixed hyperparameters
lda_params = {
    'n_topics': 10,
    'n_iter': 1000,
    'random_state': 20191122  # to make results reproducible
}

models = compute_models_parallel(dtm_bg, constant_parameters=lda_params)
models

from tmtoolkit.topicmod.model_io import print_ldamodel_topic_words

model_bg = models[0][1]
print_ldamodel_topic_words(model_bg.topic_word_, vocab_bg, top_n=10)

var_params = [{'alpha': 1/(10**x)} for x in range(1, 5)]

const_params = {
    'n_iter': 500,
    'n_topics': 10,
    'random_state': 20191122  # to make results reproducible
}

models = compute_models_parallel(dtm_bg,  # smaller DTM
                                 varying_parameters=var_params,
                                 constant_parameters=const_params)
models

var_params = [{'n_topics': k, 'alpha': 1/k} for k in range(3, 31, 1)]
var_params

from tmtoolkit.topicmod import tm_lda
tm_lda.AVAILABLE_METRICS
tm_lda.DEFAULT_METRICS

from tmtoolkit.topicmod.tm_lda import evaluate_topic_models
from tmtoolkit.topicmod.evaluate import results_by_parameter

const_params = {
    'n_iter': 1000,
    'eta': 0.1,       # "eta" aka "beta"
    'random_state': 20191122  # to make results reproducible
}

eval_results = evaluate_topic_models(dtm_bg,
                                     varying_parameters=var_params,
                                     constant_parameters=const_params,
                                     return_models=True)
eval_results[:3]  # only show first three models

eval_results_by_topics = results_by_parameter(eval_results, 'n_topics')
eval_results_by_topics[:3]

from tmtoolkit.topicmod.visualize import plot_eval_results
plot_eval_results(eval_results_by_topics);
