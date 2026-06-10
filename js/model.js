// === MODELL-ARCHITEKTUR (laut Aufgabenstellung) ===
// 2 hidden Layer à 100 Neuronen, ReLU
// Output: 1 Neuron, linear
function createModel() {
  const model = tf.sequential();

  // Hidden Layer 1: 100 Neuronen, ReLU
  // inputShape: [1] weil unser Input eindimensional ist (nur x)
  model.add(tf.layers.dense({
    units: 100,
    activation: 'relu',
    inputShape: [1]
  }));

  // Hidden Layer 2: 100 Neuronen, ReLU
  model.add(tf.layers.dense({
    units: 100,
    activation: 'relu'
  }));

  // Output Layer: 1 Neuron, linear (keine Aktivierung)
  // linear, weil unsere y-Werte beliebige reelle Zahlen sind (nicht [0,1])
  model.add(tf.layers.dense({
    units: 1,
    activation: 'linear'
  }));

  // Modell kompilieren: Loss + Optimizer festlegen
  model.compile({
    optimizer: tf.train.adam(0.01),   // Adam, Learning Rate 0.01
    loss: 'meanSquaredError'          // MSE
  });

  return model;
}

// === DATEN IN TENSOREN UMWANDELN ===
function pointsToTensors(points, useNoisy) {
  const xs = points.map(p => p.x);
  const ys = points.map(p => useNoisy ? p.yNoisy : p.yClean);

  // tensor2d mit Shape [N, 1]: N Zeilen, 1 Spalte
  const xsTensor = tf.tensor2d(xs, [xs.length, 1]);
  const ysTensor = tf.tensor2d(ys, [ys.length, 1]);

  return { xs: xsTensor, ys: ysTensor };
}

async function trainModel(model, trainPoints, useNoisy, epochs, callbacks = {}) {
  const { xs, ys } = pointsToTensors(trainPoints, useNoisy);

  const history = await model.fit(xs, ys, {
    epochs: epochs,
    batchSize: 32,
    shuffle: true,
    callbacks: callbacks   // direkt durchreichen
  });

  xs.dispose();
  ys.dispose();

  return history;
}

// === LOSS BERECHNEN (für Evaluation) ===

function evaluateModel(model, points, useNoisy) {
  return tf.tidy(() => {  // tf.tidy räumt Zwischen-Tensoren automatisch auf
    const { xs, ys } = pointsToTensors(points, useNoisy);
    const preds = model.predict(xs);
    // MSE = mean((preds - ys)^2)
    const mse = tf.losses.meanSquaredError(ys, preds);
    const mseValue = mse.dataSync()[0];  // Tensor → JS-Zahl
    xs.dispose();
    ys.dispose();
    return mseValue;
  });
}

// === MODELL SPEICHERN (Download) ===

async function saveModel(model, name) {
  await model.save(`downloads://${name}`);
}

// === MODELL LADEN (von URL/Server) ===

async function loadModelFromURL(url) {
  const model = await tf.loadLayersModel(url);
  // Geladenes Modell neu kompilieren (Loss/Optimizer gehen beim Speichern verloren)
  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'meanSquaredError'
  });
  return model;
}

// === MODELL LADEN (von Datei-Input) ===

async function loadModelFromFiles(files) {
  // tf.io.browserFiles erwartet [jsonFile, ...weightFiles]
  const jsonFile = Array.from(files).find(f => f.name.endsWith('.json'));
  const weightFiles = Array.from(files).filter(f => f.name.endsWith('.bin'));

  if (!jsonFile || weightFiles.length === 0) {
    throw new Error('Bitte model.json UND weights.bin auswählen.');
  }

  const model = await tf.loadLayersModel(
    tf.io.browserFiles([jsonFile, ...weightFiles])
  );
  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'meanSquaredError'
  });
  return model;
}

// === VORHERSAGE-KURVE berechnen ===

function computePredictionCurve(model, nPoints = 200, xMin = -2, xMax = 2) {
  return tf.tidy(() => {
    const xs = [];
    for (let i = 0; i < nPoints; i++) {
      xs.push(xMin + (xMax - xMin) * i / (nPoints - 1));
    }
    const xsTensor = tf.tensor2d(xs, [nPoints, 1]);
    const preds = model.predict(xsTensor);
    const predValues = preds.dataSync();
    return xs.map((x, i) => ({ x: x, y: predValues[i] }));
  });
}
