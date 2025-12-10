const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let session = null;

// Initialiser ONNX Runtime
async function initializeONNX() {
  const statusDisplay = document.getElementById('statusDisplay');
  try {
    statusDisplay.textContent = 'Chargement du modèle...';
    statusDisplay.style.backgroundColor = '#f39c12';
    const ort = window.ort;
    session = await ort.InferenceSession.create('/image_classifier_model.onnx');
    console.log('Modèle ONNX chargé avec succès');
    statusDisplay.textContent = 'Modèle chargé avec succès ✓';
    statusDisplay.style.backgroundColor = '#27ae60';
  } catch (error) {
    console.error('Erreur lors du chargement du modèle ONNX:', error);
    statusDisplay.textContent = 'Erreur: Impossible de charger le modèle';
    statusDisplay.style.backgroundColor = '#e74c3c';
  }
}

// Initialiser le canvas avec fond noir
ctx.fillStyle = '#000';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Charger le modèle au démarrage
initializeONNX();

// Événement souris vers le bas
canvas.addEventListener('mousedown', (e) => {
  isDrawing = true;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  ctx.beginPath();
  ctx.moveTo(x, y);
});

// Événement souris en mouvement
canvas.addEventListener('mousemove', (e) => {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineTo(x, y);
  ctx.stroke();
});

// Événement souris vers le haut
canvas.addEventListener('mouseup', () => {
  isDrawing = false;
});

canvas.addEventListener('mouseleave', () => {
  isDrawing = false;
});

// Fonction pour préparer les données du canvas pour le modèle
function getCanvasData() {
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  
  // Convertir en données 28x28 normalisées (MNIST format)
  const canvas28 = document.createElement('canvas');
  canvas28.width = 28;
  canvas28.height = 28;
  const ctx28 = canvas28.getContext('2d');
  ctx28.drawImage(canvas, 0, 0, 28, 28);
  
  const imageData28 = ctx28.getImageData(0, 0, 28, 28);
  const pixels = imageData28.data;
  
  // Convertir en array normalité (0-1)
  const input = new Float32Array(28 * 28);
  for (let i = 0; i < pixels.length; i += 4) {
    // Prendre le canal rouge (les autres sont identiques pour le dessin blanc)
    input[i / 4] = pixels[i] / 255.0;
  }
  
  return input;
}

// Bouton pour reconnaître
const recognizeBtn = document.getElementById('recognizeBtn');
recognizeBtn.addEventListener('click', async () => {
  console.log('Reconnaissance en cours...');
  console.log('Session:', session);
  
  try {
    if (!session) {
      document.getElementById('resultDisplay').textContent = 'Modèle en cours de chargement...';
      return;
    }
    
    // Préparer les données
    const input = getCanvasData();
    console.log('Données préparées, taille:', input.length);
    
    // Obtenir les informations du modèle
    const inputs = session.inputNames;
    const outputs = session.outputNames;
    console.log('Entrées disponibles:', inputs);
    console.log('Sorties disponibles:', outputs);
    
    // Créer le tenseur d'entrée avec la bonne clé
    const inputKey = inputs[0];
    const inputTensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);
    
    // Effectuer l'inférence
    const runInput = {};
    runInput[inputKey] = inputTensor;
    const results = await session.run(runInput);
    
    console.log('Résultats reçus, clés:', Object.keys(results));
    
    // Récupérer les résultats avec la bonne clé
    const outputKey = outputs[0];
    const output = results[outputKey].data;
    console.log('Sortie obtenue, longueur:', output.length);
    
    // Trouver le chiffre avec la probabilité la plus élevée
    let maxProb = -1;
    let predictedDigit = -1;
    
    for (let i = 0; i < output.length; i++) {
      if (output[i] > maxProb) {
        maxProb = output[i];
        predictedDigit = i;
      }
    }
    
    console.log(`Chiffre reconnu: ${predictedDigit}, Confiance: ${(maxProb * 100).toFixed(2)}%`);
    document.getElementById('resultDisplay').textContent = `Chiffre détecté: ${predictedDigit} (Confiance: ${(maxProb * 100).toFixed(2)}%)`;
    
  } catch (error) {
    console.error('Erreur complète:', error);
    console.error('Stack:', error.stack);
    document.getElementById('resultDisplay').textContent = 'Erreur: ' + error.message;
  }
});

// Bouton pour effacer
const clearBtn = document.getElementById('clearBtn');
clearBtn.addEventListener('click', () => {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  console.log('Dessin effacé');
});
