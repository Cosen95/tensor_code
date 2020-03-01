import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { getData } from "./data";

window.onload = async () => {
  const data = getData(400);

  tfvis.render.scatterplot(
    { name: "逻辑回归训练数据" },
    {
      values: [
        data.filter(point => point.label === 1),
        data.filter(point => point.label === 0)
      ]
    }
  );

  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 1,
      inputShape: [2],
      activation: "sigmoid"
    })
  );
  model.compile({ loss: tf.losses.logLoss, optimizer: tf.train.adam(0.1) });

  const inputs = tf.tensor(data.map(point => [point.x, point.y]));
  const labels = tf.tensor(data.map(point => point.label));

  await model.fit(inputs, labels, {
    batchSize: 40,
    epochs: 50,
    callbacks: tfvis.show.fitCallbacks({ name: "训练过程" }, ["loss"])
  });
};
