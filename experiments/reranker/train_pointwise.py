import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from transformers import HfArgumentParser, TFTrainingArguments
from dataclasses import dataclass, field
import json
import os

FEATURE_DESC = {
    'input_ids': tf.io.FixedLenFeature([512], tf.int64),
    'attention_mask': tf.io.FixedLenFeature([512], tf.int64),
    'token_type_ids': tf.io.FixedLenFeature([512], tf.int64),
    'labels': tf.io.FixedLenFeature([1], tf.int64)
}


def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, FEATURE_DESC)
    input_ids = tf.cast(example['input_ids'], tf.int32)
    attention_mask = tf.cast(example['attention_mask'], tf.int32)
    token_type_ids = tf.cast(example['token_type_ids'], tf.int32)
    labels = tf.cast(example['labels'], tf.int32)
    return {'input_ids': input_ids, 'attention_mask': attention_mask,
            'token_type_ids': token_type_ids}, labels


def create_dataset(files_path, batch_size, max_steps=-1):
    dataset_files = tf.io.gfile.glob(files_path)
    raw_train_data = tf.data.TFRecordDataset(dataset_files, num_parallel_reads=None)
    parsed_train_dataset = raw_train_data.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = parsed_train_dataset.shuffle(buffer_size=1000)
    if max_steps > -1:
        max_train_size = training_args.max_steps * batch_size
        print("number of train samples:", max_train_size)
        train_dataset = train_dataset.take(max_train_size)
    return train_dataset.batch(batch_size)


def create_model(model_name, from_pt):
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=from_pt, num_labels=2)
    return model


@dataclass
class DataArguments:
    train_files: str = "/v/tomergur/convo/reranking/quac_train_t5/*.tfrecords"
    valid_files: str = "/v/tomergur/convo/reranking/quac_dev_t5/*.tfrecords"
    test_files: str = "/v/tomergur/convo/ms_marco/records_dev_exp_doc_5/*.tfrecords"
    model_name_or_path: str = "castorini/monobert-large-msmarco-finetune-only"
    checkpoint_dir: str = None
    save_best_only: bool = False
    backup_dir: str = None
    from_pt: bool = False
    early_stop: bool = False


if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments, TFTrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()
    # Create a description of the features.
    model_name_or_path = data_args.model_name_or_path
    # model_name="bert-base-uncased"
    if not os.path.exists(training_args.output_dir):
        os.mkdir(training_args.output_dir)
    with training_args.strategy.scope():
        model = create_model(model_name_or_path, data_args.from_pt)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy']
        # metrics = metrics
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=training_args.learning_rate), loss=loss,
                      metrics=metrics)
        if training_args.do_train:
            train_dataset = create_dataset(data_args.train_files, training_args.train_batch_size,
                                           training_args.max_steps)
            valid_dataset = create_dataset(data_args.valid_files, training_args.eval_batch_size,
                                           training_args.eval_steps)
            callbacks = []
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            if data_args.early_stop:
                callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", restore_best_weights=True))
            if data_args.backup_dir:
                model_backup_callback = tf.keras.callbacks.experimental.BackupAndRestore(
                    backup_dir=data_args.backup_dir)
                callbacks.append(model_backup_callback)
            if data_args.checkpoint_dir:
                model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=data_args.checkpoint_dir,
                                                                               monitor='val_loss',
                                                                               save_best_only=data_args.save_best_only,
                                                                               save_weights_only=True)
                callbacks.append(model_checkpoint_callback)
            history = model.fit(train_dataset, epochs=int(training_args.num_train_epochs),
                                validation_data=valid_dataset, verbose=1, callbacks=callbacks)
            print(history.history)
            if data_args.checkpoint_dir and data_args.save_best_only:
                model.load_weights(data_args.checkpoint_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir)
        if training_args.do_eval:
            test_files = tf.io.gfile.glob(data_args.test_files)
            raw_test_data = tf.data.TFRecordDataset(test_files, num_parallel_reads=tf.data.AUTOTUNE)
            test_dataset = raw_test_data.map(_parse_function)
            print("eval model!!!")
            if training_args.eval_steps > -1:
                max_eval_samples = training_args.eval_steps * training_args.eval_batch_size
                test_dataset = test_dataset.take(max_eval_samples)
            res = model.evaluate(test_dataset.batch(training_args.eval_batch_size), return_dict=True)
            print("eval res:", res)
            with open("{}/{}_eval_res.json".format(training_args.output_dir, training_args.run_name), 'w') as f:
                json.dump(res, f)
