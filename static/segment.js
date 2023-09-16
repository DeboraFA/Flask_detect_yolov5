// Este exemplo usa o Fetch API para enviar a solicitação POST para a rota /segment
document.querySelector('form').addEventListener('submit', async (event) => {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const response = await fetch('/segment', {
      method: 'POST',
      body: formData
    });
  
    const data = await response.json();
    console.log(data);  // Certifique-se de verificar o que foi retornado pelo servidor
  });
  