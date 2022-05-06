import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from transformers import HfArgumentParser, TFTrainingArguments
from dataclasses import dataclass, field
import json
import os

from experiments.reranker.consistency_trainer import ConsistencyTrainer

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


FEATURE_PAIR_DESC = {
    'input_ids': tf.io.FixedLenFeature([512], tf.int64),
    'attention_mask': tf.io.FixedLenFeature([512], tf.int64),
    'token_type_ids': tf.io.FixedLenFeature([512], tf.int64),
    'input_ids_2': tf.io.FixedLenFeature([512], tf.int64),
    'attention_mask_2': tf.io.FixedLenFeature([512], tf.int64),
    'token_type_ids_2': tf.io.FixedLenFeature([512], tf.int64),
    'labels': tf.io.FixedLenFeature([1], tf.int64)
}


def _parse_pair_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, FEATURE_PAIR_DESC)
    input_ids = tf.cast(example['input_ids'], tf.int32)
    attention_mask = tf.cast(example['attention_mask'], tf.int32)
    token_type_ids = tf.cast(example['token_type_ids'], tf.int32)
    input_ids2 = tf.cast(example['input_ids_2'], tf.int32)
    attention_mask2 = tf.cast(example['attention_mask_2'], tf.int32)
    token_type_ids2 = tf.cast(example['token_type_ids_2'], tf.int32)
    labels = tf.cast(example['labels'], tf.int32)
    return {'input_ids': input_ids, 'attention_mask': attention_mask,
            'token_type_ids': token_type_ids, 'input_ids_2': input_ids2, 'attention_mask_2': attention_mask2,
            'token_type_ids_2': token_type_ids2}, labels


def create_dataset(files_path, batch_size, max_steps=-1, parse_func=_parse_function):
    dataset_files = tf.io.gfile.glob(files_path)
    raw_train_data = tf.data.TFRecordDataset(dataset_files, num_parallel_reads=None)
    parsed_train_dataset = raw_train_data.map(parse_func, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = parsed_train_dataset.shuffle(buffer_size=1000)
    #train_dataset = parsed_train_dataset
    if max_steps > -1:
        max_train_size = max_steps * batch_size
        print("number of train samples:", max_train_size)
        train_dataset = train_dataset.take(max_train_size)
    return train_dataset.batch(batch_size)


def create_dataset_interleaved(files_paths, batch_size, max_steps=-1, block_length=8):
    dataset_files = []
    for files_path in files_paths:
        dataset_files = dataset_files + tf.io.gfile.glob(files_path)
    print(dataset_files)
    files_ds = tf.data.Dataset.from_tensor_slices(dataset_files)
    parsed_train_dataset = files_ds.interleave(map_func=lambda x: tf.data.TFRecordDataset(x).map(_parse_function),
                                               num_parallel_calls=tf.data.AUTOTUNE,
                                               cycle_length=len(dataset_files), block_length=block_length)
    # train_dataset = parsed_train_dataset.shuffle(buffer_size=1000)
    train_dataset = parsed_train_dataset
    if max_steps > -1:
        max_train_size = max_steps * batch_size
        print("number of train samples:", max_train_size)
        train_dataset = train_dataset.take(max_train_size)
    return train_dataset.batch(batch_size)


def create_model(model_name, data_args):
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=data_args.from_pt, num_labels=2)
    if data_args.consistency_loss_weight == 0:
        return model
    teacher = None if data_args.manual_loss_weight > 0 or data_args.use_const_loss else TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=data_args.from_pt,
                                                                   num_labels=2)
    return ConsistencyTrainer(model, teacher)


@dataclass
class DataArguments:
    train_files: str = "/v/tomergur/convo/reranking/qrecc_train_all/*.tfrecords"
    sup_train_files: str = None
    valid_files: str = "/v/tomergur/convo/reranking/quac_dev_all/*.tfrecords"
    test_files: str = "/v/tomergur/convo/ms_marco/records_dev_exp_doc_5/*.tfrecords"
    model_name_or_path: str = "castorini/monobert-large-msmarco-finetune-only"
    info_dir: str = "/v/tomergur/convo/reranking/models/exprs/"
    checkpoint_dir: str = None
    save_best_only: bool = False
    backup_dir: str = None
    from_pt: bool = False
    consistency_loss_weight: float = 0
    manual_loss_weight: float = 0
    use_teacher: bool = False
    early_stop: bool = False
    use_const_loss:bool = False


if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments, TFTrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()
    # Create a description of the features.
    model_name_or_path = data_args.model_name_or_path
    # model_name="bert-base-uncased"
    run_name = training_args.run_name.split("/")[-2] if "/" in training_args.run_name else training_args.run_name
    if not os.path.exists(training_args.output_dir):
        os.mkdir(training_args.output_dir)
    if data_args.info_dir and not os.path.exists(data_args.info_dir):
        os.mkdir(data_args.info_dir)
    info_expr_dir = "{}/{}".format(data_args.info_dir, run_name)
    if not os.path.exists(info_expr_dir):
        os.mkdir(info_expr_dir)
    with open("{}/{}".format(info_expr_dir, "training_args.json"), 'w') as f:
        json.dump({k: v for k, v in training_args.__dict__.items() if isinstance(v, int) or isinstance(v, str)}, f,
                  indent=True)
    with open("{}/{}".format(info_expr_dir, "data_args.json"), 'w') as f:
        json.dump(data_args.__dict__, f, indent=True)
    use_pair_inputs = data_args.consistency_loss_weight > 0
    with training_args.strategy.scope():
        model = create_model(model_name_or_path, data_args)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy']
        # metrics = metrics
        if use_pair_inputs:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                 reduction=tf.keras.losses.Reduction.SUM)
            consistency_loss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=training_args.learning_rate), loss=loss,
                          metrics=metrics, consistency_loss=consistency_loss,
                          consistency_loss_weight=data_args.consistency_loss_weight,
                          manual_loss_weight=data_args.manual_loss_weight,
                          global_batch_size=training_args.train_batch_size,
                          global_eval_batch_size=training_args.eval_batch_size)
        else:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=training_args.learning_rate), loss=loss,
                          metrics=metrics)
        if training_args.do_train:
            parse_func = _parse_pair_function if use_pair_inputs else _parse_function
            train_dataset = create_dataset(data_args.train_files, training_args.train_batch_size,
                                           training_args.max_steps,
                                           parse_func) if data_args.sup_train_files is None else create_dataset_interleaved(
                [data_args.train_files, data_args.sup_train_files], training_args.train_batch_size,
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
            with open("{}/{}".format(info_expr_dir, "history.json"), 'w') as f:
                json.dump(history.history, f, indent=True)
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
