import tensorflow as tf


class ConsistencyTrainer(tf.keras.Model):
    def __init__(self, model,teacher=None):
        super(ConsistencyTrainer, self).__init__()
        self.model = model
        self.teacher=model if teacher is None else teacher

    def save_pretrained(self,path):
        self.model.save_pretrained(path)
    def compile(
            self, optimizer, metrics, loss, consistency_loss, consistency_loss_weight,manual_loss_weight, temperature=1,
    global_batch_size=128,global_eval_batch_size=256):
        super(ConsistencyTrainer, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = loss
        self.distillation_loss_fn = consistency_loss
        self.temperature = temperature
        self.cosistency_loss_weight = consistency_loss_weight
        self.manual_loss_weight=manual_loss_weight
        self.global_batch_size=global_batch_size
        self.global_eval_batch_size=global_eval_batch_size

    def train_step(self, data):
        # Since our dataset is a zip of two independent datasets,
        # after initially parsing them, we segregate the
        # respective images and labels next.
        inputs, labels = data
        cosist_inputs = {"input_ids": inputs["input_ids_2"], "attention_mask": inputs["attention_mask_2"],
                         "token_type_ids": inputs["token_type_ids_2"]}
        raw_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"],
                      "token_type_ids": inputs["token_type_ids"]}

        # Forward pass of teacher
        calc_manual_grad=self.manual_loss_weight>0
        with tf.GradientTape() as tape:
            # Forward pass of student
            teacher_predictions = self.teacher(**cosist_inputs, training=calc_manual_grad).logits
            model_predictions = self.model(**raw_inputs, training=True).logits
            # Compute losses
            loss = self.student_loss_fn(labels, model_predictions)*(1./self.global_batch_size)
            teacher_loss=self.student_loss_fn(labels, teacher_predictions)*(1./self.global_batch_size)
            consistency_predictions=tf.identity(teacher_predictions)
            consistency_loss = self.distillation_loss_fn(
                tf.nn.softmax(model_predictions / self.temperature, axis=1),
                tf.nn.softmax(consistency_predictions / self.temperature, axis=1),
            )*(1./self.global_batch_size)
            total_loss = (loss + self.cosistency_loss_weight * consistency_loss+self.manual_loss_weight*teacher_loss)

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`
        self.compiled_metrics.update_state(
            labels, tf.nn.softmax(model_predictions, axis=1)
        )

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        results.update({"teacher_loss": teacher_loss})
        results.update({"consistency_loss": consistency_loss})
        results.update({"total_loss": total_loss})
        return results


    def test_step(self, data):
        # During inference, we only pass a dataset consisting images and labels.
        x, y = data

        # Compute predictions
        y_prediction = self.model(**x, training=False)

        # Update the metrics
        model_pred=tf.nn.softmax(y_prediction.logits, axis=1)

        self.compiled_metrics.update_state(y,model_pred )
        loss = self.student_loss_fn(y, model_pred) * (1. / self.global_eval_batch_size)
        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results
