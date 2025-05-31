// main.js
document.addEventListener('DOMContentLoaded', () => {
  const videoElement = document.getElementById('video');
  const gestureDiv = document.getElementById('gestureResult');

  function updateGestureDisplay(letter, gestureSymbol, error) {
      if (error) {
          gestureDiv.textContent = `Error: ${error}`; // Tampilkan pesan error agar lebih jelas
          gestureDiv.classList.remove('visible');
      } else if (letter && letter !== '-') {
          gestureDiv.textContent = `${letter} ${gestureSymbol || ''}`.trim();
          gestureDiv.classList.add('visible');
      } else {
          gestureDiv.textContent = "-";
          gestureDiv.classList.remove('visible');
      }

      if (letter && letter !== '-' && !error) {
          gestureDiv.style.animation = 'none';
          void gestureDiv.offsetWidth; 
          gestureDiv.style.animation = 'fadeInZoom 0.6s ease forwards';
      }
  }

  async function captureAndPredict() {
      try {
          if (!videoElement) {
              console.error("MAIN.JS: Elemen video tidak ditemukan saat captureAndPredict.");
              updateGestureDisplay(null, null, "Video element missing");
              return;
          }
           if (!gestureDiv) { // Tambahkan pengecekan untuk gestureDiv juga
              console.error("MAIN.JS: Elemen gestureResult tidak ditemukan saat captureAndPredict.");
              // Mungkin tidak perlu updateGestureDisplay di sini jika gestureDiv itu sendiri null
              return;
          }


          console.log("MAIN.JS: Mencoba mengakses kamera...");
          const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
          console.log("MAIN.JS: Stream kamera didapatkan.");
          videoElement.srcObject = stream;
          
          videoElement.onloadedmetadata = async () => { // Menunggu metadata video (dimensi, dll.)
              console.log("MAIN.JS: Metadata video dimuat.");
              try {
                  await videoElement.play();
                  console.log("MAIN.JS: Video berhasil play.");
              } catch (playError) {
                  console.error("MAIN.JS: Gagal play video:", playError);
                  updateGestureDisplay(null, null, `Video play error: ${playError.name}`);
                  return; // Hentikan jika video tidak bisa play
              }

              const canvas = document.createElement('canvas');
              const ctx = canvas.getContext('2d');

              setInterval(async () => {
                  if (videoElement.paused || videoElement.ended || videoElement.readyState < videoElement.HAVE_ENOUGH_DATA) {
                      // console.log("MAIN.JS: Video tidak siap untuk di-capture atau stream berakhir.");
                      return; 
                  }

                  const captureWidth = videoElement.videoWidth;
                  const captureHeight = videoElement.videoHeight;

                  if (captureWidth === 0 || captureHeight === 0) {
                      // console.log("MAIN.JS: Dimensi video nol, skip frame.");
                      return;
                  }

                  canvas.width = captureWidth;
                  canvas.height = captureHeight;
                  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

                  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));
                  const formData = new FormData();
                  formData.append('image', blob, 'frame.jpg');

                  try {
                      const response = await fetch('/predict', { method: 'POST', body: formData });
                      if (!response.ok) {
                          const errorData = await response.json().catch(() => ({ error: `HTTP error ${response.status}` }));
                          console.error("Error from server:", errorData.error);
                          // Gunakan errorData.error di sini
                          updateGestureDisplay(null, null, errorData.error || "Prediction failed");
                          return;
                      }
                      const data = await response.json();

                      if (data.error && !data.class) {
                          updateGestureDisplay(null, data.gesture, data.error);
                      } else if (data.class) {
                          updateGestureDisplay(data.class, data.gesture, null);
                      } else {
                          updateGestureDisplay('-', data.gesture, null);
                      }
                  } catch (networkError) {
                      console.error("Network or prediction error:", networkError);
                      updateGestureDisplay(null, null, "Network error");
                  }
              }, 1000);
          };

          videoElement.onerror = (e) => { // Menangani error pada elemen video itu sendiri
              console.error("MAIN.JS: Video element error:", e);
              updateGestureDisplay(null, null, "Video element error");
          };

      } catch (err) {
          console.error("MAIN.JS: Error accessing camera: ", err.name, err.message);
          updateGestureDisplay(null, null, `Camera access error: ${err.name}`);
      }
  }

  if (videoElement && gestureDiv) {
      captureAndPredict();
  } else {
      console.error("Elemen #video atau #gestureResult tidak ditemukan saat inisialisasi akhir main.js");
      const gd = document.getElementById('gestureResult');
      if (gd) {
          gd.textContent = "Init Error (JS)";
          gd.classList.add('visible');
      }
  }
});
