import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

window.onload = () => {
  const heights = [163, 175, 186];
  const weights = [90, 120, 160];

  tfvis.render.scatterplot(
    { name: "身高体重训练数据" },
    { values: heights.map((x, i) => ({ x, y: weights[i] })) },
    { xAxisDomain: [150, 190], yAxisDomain: [80, 180] }
  );

  const inputs = tf
    .tensor(heights)
    .sub(163)
    .div(23);
  inputs.print();

  const labels = tf
    .tensor(weights)
    .sub(90)
    .div(70);
  labels.print();
};
