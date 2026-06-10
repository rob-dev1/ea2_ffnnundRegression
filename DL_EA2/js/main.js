
let currentDataset = null;   // der aktuell geladene/erzeugte Datensatz
let chartR1Clean = null;     // Referenz auf das linke R1-Chart
let chartR1Noisy = null;     // Referenz auf das rechte R1-Chart


async function init() {
  // Warten bis TFJS initialisiert hat
  await tf.ready();
  console.log('TFJS', tf.version.tfjs, '· Backend:', tf.getBackend());

  // Standard-Datensatz vom Server laden (Abgabe-Verhalten)
  try {
    currentDataset = await loadDatasetFromURL('data/dataset.json');
    console.log('Standard-Datensatz geladen:', currentDataset.meta);
  } catch (err) {
    console.error('Laden fehlgeschlagen:', err.message);
    setStatus('Fehler beim Laden des Datensatzes: ' + err.message);
    return;
  }

  // R1-Plots zeichnen
  renderR1(currentDataset);
  // === Drei vortrainierte Modelle laden ===
  let cleanModel, bestModel, overfitModel;
  try {
    cleanModel   = await loadModelFromURL('models/clean/model-200ep-clean.json');
    bestModel    = await loadModelFromURL('models/best/model-200ep-noisy.json');
    overfitModel = await loadModelFromURL('models/overfit/model-10000ep-noisy.json');
    console.log('Alle drei Modelle geladen.');
  } catch (err) {
    console.error('Modell-Laden fehlgeschlagen:', err.message);
    setStatus('Fehler beim Laden der Modelle: ' + err.message);
    return;
  }

  // R2: Clean-Modell auf unverrauschten Daten
  renderPredictionRow('r2', cleanModel, currentDataset, false);
  // R3: Best-Fit-Modell auf verrauschten Daten
  renderPredictionRow('r3', bestModel, currentDataset, true);
  // R4: Overfit-Modell auf verrauschten Daten
  renderPredictionRow('r4', overfitModel, currentDataset, true);

  // Datei-Verwaltung + Training (Buttons) aktivieren
  setupDevPanel();

  setStatus(`Datensatz geladen — ${currentDataset.train.length} Train / ${currentDataset.test.length} Test`);
}


function setStatus(text) {
  document.getElementById('status').textContent = text;
}



function renderR1(dataset) {
  // Punkte ins {x, y}-Format bringen (Chart.js Scatter braucht das)
  // useNoisy = true → nimm yNoisy, sonst yClean
  function toXY(points, useNoisy) {
    return points.map(p => ({ x: p.x, y: useNoisy ? p.yNoisy : p.yClean }));
  }

  // Alte Charts zerstören, sonst überlagern sie sich beim Neu-Zeichnen
  if (chartR1Clean) chartR1Clean.destroy();
  if (chartR1Noisy) chartR1Noisy.destroy();

  // Links: ohne Rauschen
  chartR1Clean = plotDataset(
    'chart-r1-clean',
    toXY(dataset.train, false),
    toXY(dataset.test, false)
  );

  // Rechts: mit Rauschen
  chartR1Noisy = plotDataset(
    'chart-r1-noisy',
    toXY(dataset.train, true),
    toXY(dataset.test, true)
  );
}

// === Vorhersage-Zeile rendern (R2, R3 oder R4) ===

function renderPredictionRow(prefix, model, dataset, useNoisy) {
  // Vorhersage-Kurve einmal berechnen (gleich für Train- und Test-Plot)
  const curve = computePredictionCurve(model);

  // Trainingsdaten-Plot (blaue Punkte)
  plotPrediction(`chart-${prefix}-train`, dataset.train, curve, useNoisy, 'Training', '#1f77b4');
  const trainMSE = evaluateModel(model, dataset.train, useNoisy);
  document.getElementById(`mse-${prefix}-train`).textContent =
    `MSE (Training): ${trainMSE.toFixed(4)}`;

  // Testdaten-Plot (orange Punkte)
  plotPrediction(`chart-${prefix}-test`, dataset.test, curve, useNoisy, 'Test', '#ff7f0e');
  const testMSE = evaluateModel(model, dataset.test, useNoisy);
  document.getElementById(`mse-${prefix}-test`).textContent =
    `MSE (Test): ${testMSE.toFixed(4)}`;
}

function setupDevPanel() {
  const btnGenerate = document.getElementById('btn-generate');
  const btnSave     = document.getElementById('btn-save');
  const inputLoad   = document.getElementById('input-load');
  const feedback    = document.getElementById('dev-feedback');
    // === Modell-Training ===
  const btnTrain = document.getElementById('btn-train');
  const inputEpochs = document.getElementById('input-epochs');
  const inputNoisy = document.getElementById('input-noisy');
  const trainProgress = document.getElementById('train-progress');

  function setFeedback(type, msg) {
    feedback.className = 'dev-feedback ' + type;
    feedback.textContent = msg;
  }

  // --- Generieren ---
  btnGenerate.addEventListener('click', () => {
    currentDataset = generateDataset(); //generateDataset() ist in data.js definiert
    renderR1(currentDataset);
    setFeedback('success',
      `Neuer Datensatz erzeugt (${currentDataset.train.length} Train / ${currentDataset.test.length} Test). Plots aktualisiert.`);
  });

  // --- Speichern (Download) ---
  btnSave.addEventListener('click', () => {
    if (!currentDataset) {
      setFeedback('error', 'Kein Datensatz zum Speichern vorhanden.');
      return;
    }
    downloadDataset(currentDataset, 'dataset.json'); //downloadDataset() ist in data.js definiert
    setFeedback('info', 'Datensatz wird als dataset.json heruntergeladen.');
  });

  // --- Laden (von Datei) ---
  inputLoad.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      currentDataset = await loadDatasetFromFile(file); //loadDatasetFromFile() ist in data.js definiert
      renderR1(currentDataset);
      setFeedback('success',
        `Datensatz geladen: ${file.name} (${currentDataset.train.length} Train / ${currentDataset.test.length} Test).`);
    } catch (err) {
      setFeedback('error', err.message);
    }
    inputLoad.value = '';  // erlaubt, dieselbe Datei erneut zu laden
  });

  // Standard-Datensatz ist beim Start schon da → Speichern direkt erlauben
  btnSave.disabled = false;

   btnTrain.addEventListener('click', async () => {
    if (!currentDataset) {
      setFeedback('error', 'Kein Datensatz geladen.');
      return;
    }

    const epochs = parseInt(inputEpochs.value, 10);
    const useNoisy = inputNoisy.checked;

    btnTrain.disabled = true;
    setFeedback('info', `Training läuft (${epochs} Epochen, ${useNoisy ? 'verrauscht' : 'unverrauscht'})…`);

    // Neues Modell, frisch trainieren
    const model = createModel();

    // TF Visor: Live-Loss-Kurve
    const surface = tfvis.visor().surface({ name: 'Trainings-Loss', tab: 'Training' });
    const lossValues = [];

    await trainModel(model, currentDataset.train, useNoisy, epochs, {
      onEpochEnd: (epoch, logs) => {
        lossValues.push({ x: epoch, y: logs.loss });
        // Visor nur alle paar Epochen aktualisieren (Performance bei vielen Epochen)
if (epoch % 10 === 0 || epoch === epochs - 1) {
          tfvis.render.linechart(surface, { values: lossValues },
            { xLabel: 'Epoche', yLabel: 'Loss (MSE)', width: 400, height: 300 });
          trainProgress.textContent = `Epoche ${epoch + 1}/${epochs} — Loss: ${logs.loss.toFixed(4)}`;
        }
      }
    });

    // Loss auf Train und Test berichten (ehrlich, nur zur Info)
    const trainLoss = evaluateModel(model, currentDataset.train, useNoisy);
    const testLoss = evaluateModel(model, currentDataset.test, useNoisy);
    trainProgress.textContent =
      `Fertig — Train-MSE: ${trainLoss.toFixed(4)} · Test-MSE: ${testLoss.toFixed(4)}`;

    // Modell speichern (Download)
    const suffix = useNoisy ? 'noisy' : 'clean';
    const filename = `model-${epochs}ep-${suffix}`;
    await saveModel(model, filename);

    setFeedback('success',
      `Modell trainiert & gespeichert: ${filename} (Train ${trainLoss.toFixed(4)} / Test ${testLoss.toFixed(4)}).`);

    model.dispose();
    btnTrain.disabled = false;
  });
}



init();