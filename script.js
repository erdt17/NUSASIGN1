let model;
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const predictionText = document.getElementById('prediction');
const loading = document.getElementById('loading');

// Load Graph Model (.json dari folder web_model)
async function loadModel() {
  model = await tf.loadGraphModel('web_model/model.json');
  loading.innerText = '✅ Model loaded. Showing prediction...';
}
loadModel();

// Setup webcam
async function setupWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  webcam.srcObject = stream;
  return new Promise(resolve => webcam.onloadedmetadata = () => resolve());
}

// Prediction loop
async function predictLoop() {
  if (!model) return;

  // Ambil frame webcam dan ubah ke tensor
  const tfImg = tf.browser.fromPixels(webcam)
    .resizeBilinear([224, 224])
    .expandDims(0)
    .toFloat()
    .div(255);

  // Jalankan prediksi (gunakan nama tensor input dari signature)
  const prediction = await model.executeAsync({ 'input_layer': tfImg });

  // Ambil data hasil prediksi
  const data = prediction.dataSync();
  const maxIndex = data.indexOf(Math.max(...data));

  // Ambil label dari label_map.json
  const label = await fetch('web_model/label_map.json').then(r => r.json());
  const gesture = Object.keys(label).find(key => label[key] === maxIndex);
  predictionText.innerText = gesture || '-';

  // Bersihkan memori
  tfImg.dispose();
  prediction.dispose();

  // Ulangi terus
  requestAnimationFrame(predictLoop);
}

// Jalankan webcam dan mulai prediksi
setupWebcam().then(() => predictLoop());
