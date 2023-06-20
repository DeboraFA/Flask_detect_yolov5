var cameraStream = null;
var video = document.getElementById('video');
var recognitionRunning = false;

function sendImage(imageData) {
  // Envia a imagem para o servidor Flask via AJAX
  $.ajax({
    url: '/detect',
    type: 'POST',
    contentType: 'application/json',
    data: JSON.stringify({ image: imageData }),
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
  var image_data = canvas.toDataURL('image/jpeg').split(',')[1];

  // Chama a função sendImage para enviar a imagem para o servidor Flask
  sendImage(image_data);

  // Chama a função detect novamente em loop infinito
  requestAnimationFrame(detect);
}

function startCamera() {
  if (cameraStream === null) {
    var ligarButton = document.getElementById('ligarCamera');
    ligarButton.disabled = true;
    var desligarButton = document.getElementById('desligarCamera');
    desligarButton.disabled = false;
    navigator.mediaDevices
      .getUserMedia({ video: true })
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

    // Desativa a câmera e para todos os seus tracks
    cameraStream.getTracks().forEach(function (track) {
      track.stop();
    });

    // Desativa a gravação de vídeo (luz da câmera)
    var mediaStreamTrack = cameraStream.getVideoTracks()[0];
    mediaStreamTrack.stop();

    cameraStream = null;
  }
}
