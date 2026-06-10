// === SCATTER-PLOT: Train- + Testdaten in einem Diagramm ===
// canvasId: id des <canvas>-Elements
// trainData/testData: Arrays von {x, y}
// title: optionaler Titel über dem Plot
function plotDataset(canvasId, trainData, testData, title = '') {
  const canvas = document.getElementById(canvasId);

  return new Chart(canvas, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Training',
          data: trainData,            // Format: [{x: ..., y: ...}, ...]
          backgroundColor: '#1f77b4', // blau
          pointRadius: 4
        },
        {
          label: 'Test',
          data: testData,
          backgroundColor: '#ff7f0e', // orange
          pointRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: { display: !!title, text: title }, // Titel nur anzeigen, wenn title-String nicht leer ist
        legend: { position: 'top' }
      },
      scales: {
        x: {
          type: 'linear', 
          title: { display: true, text: 'x' }, 
          min: -2.2, // etwas größer als 2 damit am Rand nicht abgeschnitten wird
          max: 2.2
        },
        y: {
          title: { display: true, text: 'y' }
        }
      }
    }
  });
}

// === VORHERSAGE-PLOT: Datenpunkte + Vorhersage-Kurve in einem Diagramm ===
// canvasId: id des <canvas>
// dataPoints: Array von {x, yClean, yNoisy}
// predictionCurve: Array von {x, y} — die Modell-Vorhersage über [-2, 2]
// useNoisy: Datenpunkte verrauscht (true) oder sauber (false) anzeigen
// pointLabel/pointColor: Beschriftung und Farbe der Punkte
function plotPrediction(canvasId, dataPoints, predictionCurve, useNoisy, pointLabel, pointColor) {
  const canvas = document.getElementById(canvasId);

  const dataXY = dataPoints.map(p => ({ x: p.x, y: useNoisy ? p.yNoisy : p.yClean }));

  return new Chart(canvas, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: pointLabel,
          data: dataXY,
          backgroundColor: pointColor,
          pointRadius: 4,
          order: 2          // Punkte über der Linie zeichnen
        },
        {
          label: 'Vorhersage y(x)',
          data: predictionCurve,
          type: 'line',
          borderColor: '#d62728',  // rot
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
          order: 1          // Linie zuerst (darunter)
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top' }
      },
      scales: {
        x: { type: 'linear', title: { display: true, text: 'x' }, min: -2.2, max: 2.2 },
        y: { title: { display: true, text: 'y' } }
      }
    }
  });
}