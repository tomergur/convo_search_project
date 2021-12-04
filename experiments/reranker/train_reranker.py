import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from transformers import HfArgumentParser, TFTrainingArguments
from dataclasses import dataclass, field
import json
import os


# region Helper classes
#TODO: handle crappy chekpointing mechanism
class WeightCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self,checkpoint_path,save_freq,initial_batch_number=0):
        self.checkpoint_path=checkpoint_path
        self.save_freq=save_freq
        self.initial_batch_number=initial_batch_number
    def on_train_batch_end(self,batch,logs):
        cur_batch=batch+1+self.initial_batch_number
        if (cur_batch)%self.save_freq!=0:
            return
        file_path=self.checkpoint_path.format(batch=cur_batch)
        self.model.save_weights(file_path)

class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        # self.model.save_weights("{}/tpu-model.h5".format(self.output_dir))
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
    input_ids = tf.cast(example['input_ids'], tf.int32)
    attention_mask = tf.cast(example['input_ids'], tf.int32)
    attention_mask = tf.cast(example['attention_mask'], tf.int32)
    token_type_ids = tf.cast(example['token_type_ids'], tf.int32)
    labels = tf.cast(example['labels'], tf.int32)
    return {'input_ids': input_ids, 'attention_mask': attention_mask,
            'token_type_ids': token_type_ids}, labels


def create_training_dataset(data_args, training_args):
    train_files = tf.io.gfile.glob(data_args.train_files)
    raw_train_data = tf.data.TFRecordDataset(train_files, num_parallel_reads=tf.data.AUTOTUNE)
    parsed_train_dataset = raw_train_data.map(_parse_function)
    train_dataset = parsed_train_dataset
    # train_dataset = parsed_train_dataset.shuffle(buffer_size=len(parsed_train_dataset))
    if training_args.max_steps > -1:
        max_train_size = training_args.max_steps * training_args.train_batch_size
        print("number of train samples:", max_train_size)
        train_dataset = train_dataset.take(max_train_size)
    if data_args.checkpoint_step>0:
        train_dataset=train_dataset.skip(data_args.checkpoint_step * training_args.train_batch_size)
    train_dataset = train_dataset.batch(training_args.train_batch_size)
    return train_dataset


def create_model(model_name,from_pt):
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, type_vocab_size=2, from_pt=from_pt)
    return model


@dataclass
class DataArguments:
    train_files: str = "/v/tomergur/convo/ms_marco/records_dev_only_q/*.tfrecords"
    test_files: str = "/v/tomergur/convo/ms_marco/records_dev_only_q/*.tfrecords"
    model_name: str="castorini/monobert-large-msmarco-finetune-only"
    checkpoint_dir: str =None
    checkpoint_step: int = 0
    from_pt:bool= False



if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments, TFTrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()
    # Create a description of the features.
    model_name = data_args.model_name
    # model_name="bert-base-uncased"
    if not os.path.exists(training_args.output_dir):
        os.mkdir(training_args.output_dir)
    with training_args.strategy.scope():
        # validation_dataset = parsed_train_dataset.skip(max_train_size)
        model = create_model(model_name,data_args.from_pt)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy']
        # metrics = metrics
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=training_args.learning_rate), loss=loss,
                      metrics=metrics)
        if training_args.do_train:
            train_dataset = create_training_dataset(data_args, training_args)
            #callbacks = [SavePretrainedCallback(output_dir=training_args.output_dir)]
            callbacks = []
            if data_args.checkpoint_dir:
                checkpoint_filepath=data_args.checkpoint_dir+"/weights_{batch}.h5"
                #model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,save_freq=training_args.save_steps,save_weights_only=True)
                model_checkpoint_callback=WeightCheckpointCallback(checkpoint_filepath,training_args.save_steps,data_args.checkpoint_step)
                callbacks.append(model_checkpoint_callback)
            if data_args.checkpoint_step>0:
                print("restore chekpoint")
                model.load_weights("{}/weights_{}.h5".format(data_args.checkpoint_dir,data_args.checkpoint_step))
            if training_args.max_steps>data_args.checkpoint_step:
                history = model.fit(train_dataset, epochs=int(training_args.num_train_epochs), verbose=1,
                                callbacks=callbacks)
                print(history.history)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir)

        if training_args.do_eval:
            test_files = tf.io.gfile.glob(data_args.test_files)
            raw_test_data = tf.data.TFRecordDataset(test_files, num_parallel_reads=tf.data.AUTOTUNE)
            test_dataset = raw_test_data.map(_parse_function)
            print("eval model!!!")
            # training_args.eval_steps
            # test_dataset.take(10 * training_args.eval_batch_size)
            if training_args.eval_steps > -1:
                max_eval_samples = training_args.eval_steps * training_args.eval_batch_size
                test_dataset = test_dataset.take(max_eval_samples)
            res = model.evaluate(test_dataset.batch(training_args.eval_batch_size), return_dict=True)
            print("eval res:", res)
            with open("{}/{}_eval_res.json".format(training_args.output_dir, training_args.run_name), 'w') as f:
                json.dump(res, f)
