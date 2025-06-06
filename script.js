let model;
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const predictionText = document.getElementById('prediction');
const loading = document.getElementById('loading');

// Load Graph Model (.json dari TensorFlow.js SavedModel)
async function loadModel() {
  model = await tf.loadGraphModel('web_model/model.json'); // folder hasil konversi
  loading.innerText = '✅ Model loaded. Showing prediction...';
}
loadModel();

// Webcam setup
async function setupWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  webcam.srcObject = stream;
  return new Promise(resolve => webcam.onloadedmetadata = () => resolve());
}

// Predict loop for Graph Model
async function predictLoop() {
  if (!model) return;

  // Ambil frame dari webcam, ubah ke tensor
  const tfImg = tf.browser.fromPixels(webcam)
    .resizeBilinear([96, 96])
    .expandDims(0)
    .toFloat()
    .div(255);

  // Graph Model input/output biasanya pakai nama tensor, cek model signature jika error
  const prediction = await model.executeAsync({'serving_default_input_1': tfImg}); // <--- GANTI sesuai input tensor

  const data = prediction.dataSync(); // hasil array of probability
  const maxIndex = data.indexOf(Math.max(...data));

  // Ambil label
  const label = await fetch('web_model/label_map.json').then(r => r.json());
  const gesture = Object.keys(label).find(key => label[key] === maxIndex);
  predictionText.innerText = gesture || '-';

  tfImg.dispose();
  prediction.dispose();
  requestAnimationFrame(predictLoop);
}

setupWebcam().then(() => predictLoop());
