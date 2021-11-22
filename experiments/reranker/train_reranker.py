import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification
from transformers import HfArgumentParser, TFTrainingArguments
from dataclasses import dataclass, field

# region Helper classes
class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_pretrained(self.output_dir)


FEATURE_DESC = {
    'input_ids': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    'attention_mask': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    'token_type_ids': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    'labels': tf.io.FixedLenFeature([], tf.int64)
}


def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, FEATURE_DESC)
    return {'input_ids': example['input_ids'], 'attention_mask': example['attention_mask'],
            'token_type_ids': example['token_type_ids']}, example['labels']

@dataclass
class DataArguments:
    train_file:str = "/v/tomergur/convo/ms_marco/train_pairs_q_only.tfrecords"
    test_file:str ="/v/tomergur/convo/ms_marco/dev_q_only.tfrecords"

if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments,TFTrainingArguments))
    data_args,training_args= parser.parse_args_into_dataclasses()
    # Create a description of the features.
    model_name = "castorini/monobert-large-msmarco-finetune-only"
    # model_name="bert-base-uncased"
    # put here dataset
    raw_test_data = tf.data.TFRecordDataset(data_args.test_file)
    test_dataset = raw_test_data.map(_parse_function)
    with training_args.strategy.scope():
        raw_train_data = tf.data.TFRecordDataset(data_args.train_file)
        parsed_train_dataset = raw_train_data.map(_parse_function)
        if training_args.max_steps > -1:
            max_train_size = training_args.max_steps * training_args.train_batch_size
        train_dataset = parsed_train_dataset.take(max_train_size)
        validation_dataset = parsed_train_dataset.skip(max_train_size)
        model = TFAutoModelForSequenceClassification.from_pretrained(model_name, type_vocab_size=2,from_pt=True)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy']
        # metrics = metrics
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=training_args.learning_rate), loss=loss,metrics = metrics)
        if training_args.do_train:
            callbacks = [SavePretrainedCallback(output_dir=training_args.output_dir)]
            history=model.fit(train_dataset.batch(training_args.train_batch_size),
                      validation_data=validation_dataset.batch(training_args.eval_batch_size),
                      epochs=int(training_args.num_train_epochs),validation_steps=training_args.eval_steps, verbose=1, callbacks=callbacks)
            print(history)
        if training_args.do_eval:
            res = model.evaluate(test_dataset.take(2048*500).batch(2048))
            print("eval res:", res)
