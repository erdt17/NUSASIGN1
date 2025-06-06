let model;
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const predictionText = document.getElementById('prediction');
const loading = document.getElementById('loading');

async function loadModel() {
  model = await tf.loadLayersModel('model_tfjs/model.json');
  loading.innerText = '✅ Model loaded. Showing prediction...';
}
loadModel();

async function setupWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  webcam.srcObject = stream;
  return new Promise(resolve => webcam.onloadedmetadata = () => resolve());
}

async function predictLoop() {
  if (!model) return;
  const tfImg = tf.browser.fromPixels(webcam).resizeBilinear([96, 96]).expandDims(0).toFloat().div(255);
  const prediction = await model.predict(tfImg).data();
  const maxIndex = prediction.indexOf(Math.max(...prediction));
  const label = await fetch('model_tfjs/label_map.json').then(r => r.json());
  const gesture = Object.keys(label).find(key => label[key] === maxIndex);
  predictionText.innerText = gesture || '-';
  tfImg.dispose();
  requestAnimationFrame(predictLoop);
}

setupWebcam().then(() => predictLoop());