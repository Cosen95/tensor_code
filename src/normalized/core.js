import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

window.onload = async () => {
  const heights = [163, 175, 186];
  const weights = [45, 60, 80];

  tfvis.render.scatterplot(
    { name: "身高体重训练数据" },
    { values: heights.map((x, i) => ({ x, y: weights[i] })) },
    { xAxisDomain: [150, 200], yAxisDomain: [40, 100] }
  );

  const inputs = tf
    .tensor(heights)
    .sub(163)
    .div(23);

  const labels = tf
    .tensor(weights)
    .sub(45)
    .div(35);

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.sgd(0.1)
  });

  await model.fit(inputs, labels, {
    batchSize: 3,
    epochs: 200,
    callbacks: tfvis.show.fitCallbacks(
      {
        name: "训练过程"
      },
      ["loss"]
    )
  });

  const output = model.predict(
    tf
      .tensor([193])
      .sub(163)
      .div(23)
  );
  alert(
    `如果 身高 为 193，那么预测 体重 为 ${
      output
        .mul(35)
        .add(45)
        .dataSync()[0]
    }kg`
  );
};
