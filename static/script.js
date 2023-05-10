// Obtem referências aos elementos do HTML
const video = document.querySelector('video');
const startBtn = document.querySelector('#startBtn');
const stopBtn = document.querySelector('#stopBtn');

// Objeto para armazenar o stream da câmera
let stream;

// Função para ligar a câmera
async function startCamera() {
  try {
    // Obtém o stream da câmera e define-o como fonte de vídeo
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (error) {
    console.error(error);
  }
}

// Função para desligar a câmera
function stopCamera() {
  // Para o stream da câmera
  stream.getTracks().forEach(track => track.stop());
  // Define a fonte de vídeo como vazia
  video.srcObject = null;
}

// Adiciona os eventos de clique nos botões de ligar e desligar a câmera
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);

