// === GROUND TRUTH ===
// Die "unbekannte" Funktion, die das Netz approximieren soll.
// In der Realität kennt man sie nicht — wir nutzen sie nur zum Daten erzeugen.
function groundTruth(x) {
  return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

// === GAUSSIAN NOISE (Box-Muller-Transformation) ===
// Erzeugt eine normalverteilte Zufallszahl mit Mittelwert 0 und gegebener
// Standardabweichung. Math.random() ist gleichverteilt — Box-Muller wandelt
// zwei gleichverteilte Werte in einen normalverteilten um.
function gaussianNoise(stdDev) {
  let u1 = 0, u2 = 0;
  // Math.random() liefert [0,1); wir wollen (0,1], um log(0) zu vermeiden
  while (u1 === 0) u1 = Math.random();
  while (u2 === 0) u2 = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return z * stdDev;
}

const NOISE_VARIANCE = 0.05;
const NOISE_STD = Math.sqrt(NOISE_VARIANCE);  // ≈ 0.2236

// === KONSTANTEN ===
const N_TOTAL = 100;   // Gesamtzahl Datenpunkte
const N_TRAIN = 50;    // davon Training
const N_TEST = 50;     // davon Test
const X_MIN = -2;
const X_MAX = 2;


// === DATENSATZ ERZEUGEN (A1) ===
function generateDataset() {
  // --- Schritt 1: N gleichverteilte x-Werte aus [-2, 2] ---
  const xValues = [];
  for (let i = 0; i < N_TOTAL; i++) {
    xValues.push(X_MIN + Math.random() * (X_MAX - X_MIN));
  }

  // --- Schritt 2: y(x) berechnen (unverrauscht) ---
  const points = xValues.map(x => ({
    x: x,
    yClean: groundTruth(x)
  }));

  // --- Schritt 3: zufällig mischen, dann 50/50 splitten ---
  shuffle(points);
  const trainPoints = points.slice(0, N_TRAIN);
  const testPoints  = points.slice(N_TRAIN, N_TOTAL);

  // --- Schritt 4: Rauschen addieren (nur auf y, nicht auf x) ---
  // Jeder Punkt bekommt zusätzlich yNoisy = yClean + Gaussian Noise
  for (const p of trainPoints) {
    p.yNoisy = p.yClean + gaussianNoise(NOISE_STD);
  }
  for (const p of testPoints) {
    p.yNoisy = p.yClean + gaussianNoise(NOISE_STD);
  }

  // --- Ergebnis: ein Objekt mit Train- und Test-Split ---
  return {
    meta: {
      nTotal: N_TOTAL,
      nTrain: N_TRAIN,
      nTest: N_TEST,
      xMin: X_MIN,
      xMax: X_MAX,
      noiseVariance: NOISE_VARIANCE,
      createdAt: new Date().toISOString()
    },
    train: trainPoints,
    test: testPoints
  };
}

// === HILFSFUNKTION: Array in-place mischen (Fisher-Yates) ===
function shuffle(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

// === DATENSATZ ALS JSON SPEICHERN (Download) ===
function downloadDataset(dataset, filename = 'dataset.json') {
  const json = JSON.stringify(dataset, null, 2); // null,2 = hübsch eingerückt
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  // Unsichtbaren Download-Link erzeugen und klicken
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// === DATENSATZ AUS DATEI LADEN (File-Input) ===
// Gibt ein Promise zurück, das mit dem Datensatz-Objekt auflöst.
function loadDatasetFromFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const dataset = JSON.parse(e.target.result);
        if (!validateDataset(dataset)) {
          reject(new Error('Datei hat kein gültiges Datensatz-Format.'));
          return;
        }
        resolve(dataset);
      } catch (err) {
        reject(new Error('JSON konnte nicht geparst werden: ' + err.message));
      }
    };
    reader.onerror = () => reject(new Error('Datei konnte nicht gelesen werden.'));
    reader.readAsText(file);
  });
}

// === DATENSATZ VOM SERVER LADEN (Fetch) ===
// Für die Abgabe: lädt die feste data/dataset.json beim Seitenstart.
async function loadDatasetFromURL(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Datensatz nicht gefunden (${response.status}): ${url}`);
  }
  const dataset = await response.json();
  if (!validateDataset(dataset)) {
    throw new Error('Geladener Datensatz hat kein gültiges Format.');
  }
  return dataset;
}

// === VALIDIERUNG ===
// Prüft, ob ein geladenes Objekt die erwartete Struktur hat (QA-Anforderung).
function validateDataset(ds) {
  if (!ds || typeof ds !== 'object') return false;
  if (!Array.isArray(ds.train) || !Array.isArray(ds.test)) return false;
  // Stichprobe: erster Train-Punkt muss x, yClean, yNoisy haben
  const p = ds.train[0];
  if (!p || typeof p.x !== 'number' ||
           typeof p.yClean !== 'number' ||
           typeof p.yNoisy !== 'number') {
    return false;
  }
  return true;
}