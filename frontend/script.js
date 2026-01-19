document.addEventListener('DOMContentLoaded', () => {
    // --- State & Constants ---
    const API_URL = "http://127.0.0.1:5000";
    let uploadedFile = null;

    const views = {
        home: document.getElementById('home-view'),
        upload: document.getElementById('upload-view'),
        predict: document.getElementById('predict-view'),
        learn: document.getElementById('learn-view')

    };

    // --- Utility Functions ---

    function setView(viewId) {
        Object.values(views).forEach(view => {
            view.classList.add('hidden');
        });
        const activeView = document.getElementById(viewId);
        if (activeView) {
            activeView.classList.remove('hidden');
            window.scrollTo(0, 0); // Scroll to top on view change
        }
    }

    // --- API & Logic (unchanged) ---

    const callPredictApi = async (file) => {
        const errorMsg = document.getElementById('error-message');
        const predictBtn = document.getElementById('predict-button');

        errorMsg.classList.add('hidden');
        predictBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append("file", file);

            const response = await axios.post(`${API_URL}/predict`, formData);
            return response.data;

        } catch (err) {
            const errorMessage = err.response?.data?.error || err.message || 'Network Error';
            errorMsg.textContent = 'Error: ' + errorMessage;
            errorMsg.classList.remove('hidden');
            throw err;
        } finally {
              predictBtn.textContent = "Predict";
              predictBtn.disabled = false;
            


        }
    };

    // --- Rendering Functions (unchanged, uses data from API) ---

    function renderPredictionResults(data) {
        const summary = document.getElementById('prediction-summary');
        const originalImgContainer = document.getElementById('original-image-container');
        const gradcamImgContainer = document.getElementById('gradcam-image-container');

        
        // 1. Render Images (Side-by-Side)
        originalImgContainer.innerHTML = data.original_image ? 
            `<img src="data:image/jpeg;base64,${data.original_image}" alt="Original Image" />` : 
            '<span>Image Unavailable</span>';
        
        gradcamImgContainer.innerHTML = data.gradcam_image ? 
            `<img src="data:image/jpeg;base64,${data.gradcam_image}" alt="Grad-CAM Heatmap" />` : 
            '<span>Grad-CAM Unavailable</span>';
        
        // 2. Render Summary
        summary.innerHTML = `
                <h3 class="summary-title">Prediction: <span style="font-weight:bold;">${data.prediction}</span></h3>

        `;

        // 3. Render Probabilities List
        
    }

    // --- Upload Handlers ---

    const fileInput = document.getElementById('file-upload');
    const dragDropArea = document.getElementById('drag-drop-area');


    function updatePreview(file) {
        document.getElementById('error-message').classList.add('hidden');
        
        const previewContainer = document.getElementById('image-preview-container');
        const noImageText = document.getElementById('no-image-text');
        
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedFile = file;
            if (noImageText) noImageText.classList.add('hidden');
             
                previewContainer.innerHTML = `<img src="${e.target.result}" alt="Image Preview" />`;

            
        };
        reader.readAsDataURL(file);
    }

    // --- Initialize Handlers and Routing ---

    // Event Delegation for Navigation
    document.body.addEventListener('click', (e) => {
  const target = e.target.closest('button, a, label');
  if (!target) return;

  if (target.id === 'home-start-button') {
    window.location.hash = 'upload';
    e.preventDefault();
  } else if (target.id === 'analyze-again-button') {
    window.location.hash = 'upload';
    e.preventDefault();
  } else if (target.id === 'home-learn-button') {
    window.location.hash = 'learn';
    e.preventDefault();
  } else if (target.id === 'learn-back-button') {
    window.location.hash = '';
    e.preventDefault();
  }
   else if (target.id === 'back-home-button') {
    window.location.hash = '';
    e.preventDefault();
  }
  // --- Clear upload & preview when Analyze Again is clicked ---
const analyzeAgainBtn = document.getElementById('analyze-again-button');
if (analyzeAgainBtn) {
  analyzeAgainBtn.addEventListener('click', () => {
    // Reset uploaded file reference
    uploadedFile = null;

    // Clear preview image
    const previewContainer = document.getElementById('image-preview-container');
    if (previewContainer) previewContainer.innerHTML = '';

    // Reset file input
    const fileInput = document.getElementById('file-upload');
    if (fileInput) fileInput.value = '';

    // Hide "No Image" text again if you use one
    const noImageText = document.getElementById('no-image-text');
    if (noImageText) noImageText.classList.remove('hidden');

    // Navigate cleanly to upload view
    window.location.hash = 'upload';
  });
}

});
    // File Input Change
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                updatePreview(e.target.files[0]);
            }
        });
    }

    // Drag-Drop Functionality (Attached to the interaction zone)
    if (dragDropArea) {
        // Must explicitly prevent default behavior in all drag events for drop to work
        dragDropArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            dragDropArea.classList.add('drag-over');
        });

        dragDropArea.addEventListener('dragleave', () => {
            dragDropArea.classList.remove('drag-over');
        });

        dragDropArea.addEventListener('drop', (e) => {
            e.preventDefault();
            dragDropArea.classList.remove('drag-over');
            if (e.dataTransfer.files.length) {
                updatePreview(e.dataTransfer.files[0]);
            }
        });
    }

    // Predict Button Handler
    const predictButton = document.getElementById('predict-button');
    if (predictButton) {
        predictButton.addEventListener('click', async () => {
            if (!uploadedFile) {
                alert("Please upload an image first!");
                return;
            }

            predictButton.textContent = "Predicting...";
            predictButton.disabled = true;

            
            try {
                // Ensure loading message shows if predicting takes time
                
                const results = await callPredictApi(uploadedFile);
                renderPredictionResults(results);
                window.location.hash = 'predict';
            } catch (e) {
                // Error displayed by callPredictApi
            } finally {
                // Ensure loading message is hidden after API call completes
                document.getElementById('loading-message').classList.add('hidden');
            }
        });
    }


    // Initial View Routing
    function handleHashChange() {
  const hash = window.location.hash.slice(1);
  const backHomeBtn = document.getElementById('back-home-button');

  if (hash === 'upload') {
    setView('upload-view');
    backHomeBtn.classList.add('hidden');
  } else if (hash === 'predict') {
    setView('predict-view');
    backHomeBtn.classList.remove('hidden'); // show only here
  } else if (hash === 'learn') {
    setView('learn-view');
    backHomeBtn.classList.add('hidden');
  } else {
    setView('home-view');
    backHomeBtn.classList.add('hidden');
  }
}

    window.addEventListener('hashchange', handleHashChange);
    handleHashChange(); // Run once on load
});