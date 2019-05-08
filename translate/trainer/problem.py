
import os
import tensorflow as tf
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators.translate import TranslateProblem
PROBLEM= 'translating_en_pr'
os.environ['PROBLEM'] = PROBLEM
tf.summary.FileWriterCache.clear() # ensure filewriter cache is clear for TensorBoard events file

@registry.register_problem('translating_en_pr')
class Trsanlate_Persian(text_problems.Text2TextProblem):
  

    @property
    def approx_vocab_size(self):
        return 500000  # ~8k

    @property
    def is_generate_per_split(self):
    # generate_data will NOT shard the data into TRAIN and EVAL for us.
        return False

    @property
    def dataset_splits(self):

    # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 90,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 10,
        }]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        source = open('Nueral Machine Translation/data/en/PEN.ixml')
        target = open('Nueral Machine Translation/data/fa/PEN.ixml')
        en_sens = source.read().split('\n')
        de_sens = target.read().split('\n')
        for en, de in zip(en_sens, de_sens):
            yield {
                "inputs":en,
                "targets":de
            }
            
    # Smaller than the typical translate model, and with more regularization
    @registry.register_hparams('transformer_translate_en_pr')
    def transformer_translate_en_pr():
        hparams = transformer.transformer_base()
        hparams.num_hidden_layers = 2
        hparams.hidden_size = 128
        hparams.filter_size = 512
        hparams.num_heads = 4
        hparams.attention_dropout = 0.6
        hparams.layer_prepostprocess_dropout = 0.6
        hparams.learning_rate = 0.05
        return hparams
