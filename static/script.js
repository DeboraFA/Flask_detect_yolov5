var cameraStream = null;
var video = document.getElementById('video');

  // video constraints
  const constraints = {
    video: {
      width: {
        min: 1280,
        ideal: 1920,
        max: 2560,
      },
      height: {
        min: 720,
        ideal: 1080,
        max: 1440,
      },
    },
  };


var recognitionRunning = false;
var facingMode = 'environment'; // 'user' para câmera frontal, 'environment' para câmera traseira

function sendImage(imageData) {
  // Redimensiona a imagem para 640x640 pixels
  var resizedCanvas = document.createElement('canvas');
  resizedCanvas.width = 640;
  resizedCanvas.height = 640;
  var resizedContext = resizedCanvas.getContext('2d');
  resizedContext.drawImage(video, 0, 0, 640, 640);
  var resizedImageData = resizedCanvas.toDataURL('image/png').split(',')[1];

  // Envia a imagem redimensionada para o servidor Flask via AJAX
  $.ajax({
    url: '/detect',
    type: 'POST',
    contentType: 'application/json',
    data: JSON.stringify({ image: resizedImageData }),
    success: function (response) {
      // Processa a resposta do servidor Flask
      // Você pode adicionar o código para exibir ou manipular a resposta aqui, se necessário
    },
    error: function (error) {
      console.log(error);
    }
  });
}

function detect() {
  if (!recognitionRunning) {
    return;
  }

  // Cria um canvas temporário para capturar o quadro atual do vídeo
  var canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  var context = canvas.getContext('2d');
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Converte o canvas para base64
  var image_data = canvas.toDataURL('image/png').split(',')[1];

  // Chama a função sendImage para enviar a imagem para o servidor Flask
  sendImage(image_data);

  // Chama a função detect novamente em loop infinito
  if (recognitionRunning) {
    requestAnimationFrame(detect);
  }
}

function switchCamera() {
  if (constraints.video.facingMode === 'user') {
    constraints.video.facingMode = 'environment'; // Alterna para câmera traseira
  } else {
    constraints.video.facingMode = 'user'; // Alterna para câmera frontal
  }

  stopCamera();
  startCamera();
}

function startCamera() {
  if (cameraStream === null) {
    var ligarButton = document.getElementById('ligarCamera');
    ligarButton.disabled = true;
    var desligarButton = document.getElementById('desligarCamera');
    desligarButton.disabled = false;
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: facingMode } })
      .then(function (stream) {
        video.srcObject = stream;
        video.play();
        cameraStream = stream;
        recognitionRunning = true;
        detect();
      })
      .catch(function (error) {
        console.log('Erro ao acessar a câmera:', error);
      });

    // Ajustar o tamanho do elemento de vídeo para a largura e altura da janela do navegador
    video.width = window.innerWidth;
    video.height = window.innerHeight;
  }
}

function stopCamera() {
  if (cameraStream !== null) {
    var ligarButton = document.getElementById('ligarCamera');
    ligarButton.disabled = false;
    var desligarButton = document.getElementById('desligarCamera');
    desligarButton.disabled = true;
    video.pause();
    video.srcObject = null;

    // Para a detecção definindo recognitionRunning como false
    recognitionRunning = false;

    // Desativa aGrave vídeo (luz da câmera)
    var mediaStreamTrack = cameraStream.getVideoTracks()[0];
    mediaStreamTrack.stop();

    cameraStream = null;
  }
}
