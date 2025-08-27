/**
 * War Detection Pipeline - Frontend Application
 * Main JavaScript file handling all user interactions and API communication
 */

class WarDetectionApp {
    constructor() {
        this.config = window.CONFIG;
        this.websocket = null;
        this.currentJob = null;
        this.currentFile = null;
        this.isProcessing = false;
        this.realTimeDetectionActive = false;
        this.detectionInterval = null;
        this.detectionCanvas = null;
        this.realTimeDetections = [];

        // Initialize the application
        this.init();
    }

    init() {
        console.log('Initializing War Detection Pipeline App');

        // Setup event listeners
        this.setupEventListeners();

        // Initialize WebSocket connection
        this.initWebSocket();

        // Setup drag and drop
        this.setupDragAndDrop();

        // Initialize UI components
        this.initUI();

        console.log('✅ App initialized successfully');
    }

    setupEventListeners() {
        // File input change
        document.getElementById('file-input').addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });

        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Video player time updates
        const videoPlayer = document.getElementById('video-player');
        if (videoPlayer) {
            videoPlayer.addEventListener('timeupdate', () => {
                this.onVideoTimeUpdate();
            });

            // Real-time detection on video play
            videoPlayer.addEventListener('play', () => {
                this.startRealTimeDetection();
            });

            videoPlayer.addEventListener('pause', () => {
                this.stopRealTimeDetection();
            });

            videoPlayer.addEventListener('seeked', () => {
                // Clear overlays when seeking
                this.clearDetectionOverlays();
            });
        }

        // Timeline event clicks
        document.addEventListener('click', (e) => {
            if (e.target.closest('.timeline-event')) {
                const timestamp = parseFloat(e.target.closest('.timeline-event').dataset.timestamp);
                this.seekToTimestamp(timestamp);
            }
        });
    }

    // Audio-only background analysis
    async startAudioAnalysisOnly(filename) {
        console.log('Starting audio-only analysis in background...');

        try {
            console.log(`Calling endpoint: /process_audio_only/${filename}`);

            // Use the audio-only endpoint
            const response = await fetch(`/process_audio_only/${filename}`, {
                method: 'POST'
            });

            console.log('Response status:', response.status);

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Audio analysis response error:', errorText);
                throw new Error(`Audio analysis failed: ${response.status}`);
            }

            const result = await response.json();
            console.log('Audio analysis started:', result);

            this.currentJob = result.job_id;

            // Subscribe to job updates
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({
                    type: 'subscribe_job',
                    job_id: this.currentJob
                }));
                console.log('Subscribed to audio job updates:', this.currentJob);
            }

            // Show notification
            this.showNotification('Audio analysis started in background...', 'info');

        } catch (error) {
            console.error('Audio analysis start failed:', error);
            this.showNotification('Audio analysis failed to start: ' + error.message, 'warning');
        }
    }

    setupDragAndDrop() {
        const uploadArea = document.getElementById('upload-area');

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });
    }

    initUI() {
        // Hide all sections initially
        this.hideAllSections();

        // Show upload section
        document.getElementById('upload-section').style.display = 'block';

        // Update connection status
        this.updateConnectionStatus('connecting');
    }

    hideAllSections() {
        document.getElementById('media-display').style.display = 'none';
        document.getElementById('processing-status').style.display = 'none';
        document.getElementById('detailed-results').style.display = 'none';
    }

    initWebSocket() {
        console.log('Connecting to WebSocket...');

        try {
            this.websocket = new WebSocket(this.config.wsUrl);

            this.websocket.onopen = () => {
                console.log('✅ WebSocket connected');
                this.updateConnectionStatus('connected');

                // Send ping to test connection
                this.websocket.send(JSON.stringify({
                    type: 'ping'
                }));
            };

            this.websocket.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleWebSocketMessage(message);
            };

            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus('disconnected');

                // Try to reconnect after 3 seconds
                setTimeout(() => {
                    this.initWebSocket();
                }, 3000);
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('disconnected');
            };

        } catch (error) {
            console.error('Failed to initialize WebSocket:', error);
            this.updateConnectionStatus('disconnected');
        }
    }

    updateConnectionStatus(status) {
        const statusElement = document.getElementById('ws-status');
        const statusMap = {
            'connecting': { text: 'Connecting...', class: 'connecting' },
            'connected': { text: 'Connected', class: 'connected' },
            'disconnected': { text: 'Disconnected', class: 'disconnected' }
        };

        const statusInfo = statusMap[status] || statusMap['disconnected'];
        statusElement.innerHTML = `<i class="fas fa-circle"></i> ${statusInfo.text}`;
        statusElement.className = `status-indicator ${statusInfo.class}`;
    }

    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'pong':
                console.log('WebSocket ping/pong successful');
                break;

            case 'job_update':
                this.handleJobUpdate(message.data);
                break;

            case 'subscription_confirmed':
                console.log(`Subscribed to job updates: ${message.job_id}`);
                break;

            default:
                console.log('Unknown WebSocket message:', message);
        }
    }

    async handleFileSelect(file) {
        console.log('File selected:', file.name);

        // Validate file
        if (!this.validateFile(file)) {
            return;
        }

        this.currentFile = file;

        // Show upload progress
        this.showUploadProgress();

        try {
            // Upload file
            const uploadResult = await this.uploadFile(file);
            console.log('Upload result:', uploadResult);

            // Display media
            this.displayMedia(uploadResult);

            // Handle different file types differently
            if (uploadResult.file_type === 'image') {
                // For images: Run instant detection, no background processing
                await this.processImageInstantly(uploadResult);
            } else if (uploadResult.file_type === 'video') {
                // For videos: Just display + start ONLY audio analysis in background
                this.showNotification('Video ready! Press play for real-time detection.', 'info');

                // Start only audio analysis (no object detection, no deduplication)
                await this.startAudioAnalysisOnly(uploadResult.filename);
            }

        } catch (error) {
            console.error('File handling failed:', error);
            this.showNotification('Upload failed: ' + error.message, 'error');
            this.resetToUpload();
        }
    }

    validateFile(file) {
        // Check file size
        if (file.size > this.config.maxFileSize) {
            const maxSizeMB = (this.config.maxFileSize / (1024 * 1024)).toFixed(1);
            this.showNotification(`File too large. Maximum size: ${maxSizeMB}MB`, 'error');
            return false;
        }

        // Check file type
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        if (!this.config.allowedFormats.includes(fileExtension)) {
            this.showNotification(`File type not supported. Allowed: ${this.config.allowedFormats.join(', ')}`, 'error');
            return false;
        }

        return true;
    }

    showUploadProgress() {
        document.getElementById('upload-progress').style.display = 'block';
        const progressFill = document.getElementById('upload-progress-fill');
        const progressText = document.getElementById('upload-progress-text');

        // Animate upload progress
        progressFill.style.width = '0%';
        progressText.textContent = 'Uploading...';

        // Simulate progress animation
        let progress = 0;
        const interval = setInterval(() => {
            progress += 10;
            progressFill.style.width = progress + '%';
            if (progress >= 90) {
                clearInterval(interval);
            }
        }, 100);
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const progressFill = document.getElementById('upload-progress-fill');
        const progressText = document.getElementById('upload-progress-text');

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        // Complete upload progress
        progressFill.style.width = '100%';
        progressText.textContent = 'Upload complete!';

        return await response.json();
    }

    displayMedia(uploadResult) {
        // Hide upload section
        document.getElementById('upload-section').style.display = 'none';
        document.getElementById('upload-progress').style.display = 'none';

        // Show media display (keep it visible!)
        document.getElementById('media-display').style.display = 'block';

        // Update media info
        document.getElementById('media-title').textContent = uploadResult.original_filename;
        document.getElementById('file-size').textContent = `Size: ${this.formatFileSize(uploadResult.file_size)}`;
        document.getElementById('file-type').textContent = `Type: ${uploadResult.file_type.toUpperCase()}`;

        if (uploadResult.file_type === 'video') {
            this.displayVideo(uploadResult);
        } else {
            this.displayImage(uploadResult);
        }

        // Show notification that user can play video now
        this.showNotification('Video ready! Press play to see real-time detection overlays.', 'info');
    }

    displayVideo(uploadResult) {
        const videoContainer = document.getElementById('video-container');
        const videoPlayer = document.getElementById('video-player');

        videoContainer.style.display = 'block';
        document.getElementById('image-container').style.display = 'none';

        videoPlayer.src = `/video/${uploadResult.filename}`;
        videoPlayer.load();

        // Update duration when metadata loads
        videoPlayer.addEventListener('loadedmetadata', () => {
            const duration = this.formatDuration(videoPlayer.duration);
            document.getElementById('file-duration').textContent = `Duration: ${duration}`;
        });
    }

    displayImage(uploadResult) {
        const imageContainer = document.getElementById('image-container');
        const imageDisplay = document.getElementById('image-display');

        imageContainer.style.display = 'block';
        document.getElementById('video-container').style.display = 'none';
        document.getElementById('file-duration').textContent = 'Duration: --';

        imageDisplay.src = `/image/${uploadResult.filename}`;
    }

    async startProcessing(filename) {
        console.log('Starting processing for:', filename);

        try {
            const response = await fetch(`/process/${filename}`, {
                method: 'POST'
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Processing failed to start');
            }

            const result = await response.json();
            this.currentJob = result.job_id;
            this.isProcessing = true;

            console.log('Processing started:', result);

            // Subscribe to job updates
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({
                    type: 'subscribe_job',
                    job_id: this.currentJob
                }));
            }

            // Show processing status
            this.showProcessingStatus();

        } catch (error) {
            console.error('Processing start failed:', error);
            this.showNotification('Processing failed to start: ' + error.message, 'error');
            throw error;
        }
    }

    showProcessingStatus() {
        // DON'T hide media display - keep video visible!
        // document.getElementById('media-display').style.display = 'none';

        // Show processing status as overlay or in right panel
        document.getElementById('processing-status').style.display = 'block';

        // Reset progress
        document.getElementById('processing-progress-fill').style.width = '0%';
        document.getElementById('progress-percentage').textContent = '0%';
        document.getElementById('current-step').textContent = 'Initializing...';

        // Reset all steps
        document.querySelectorAll('.step').forEach(step => {
            step.classList.remove('active', 'completed');
        });

        // Show notification that processing is running in background
        this.showNotification('Background processing started. You can play the video to see real-time detection!', 'info');
    }

    handleJobUpdate(jobStatus) {
        console.log('Job update:', jobStatus);

        if (jobStatus.job_id !== this.currentJob) {
            return; // Not our job
        }

        // Update progress
        const progressPercent = Math.round(jobStatus.progress * 100);
        document.getElementById('processing-progress-fill').style.width = progressPercent + '%';
        document.getElementById('progress-percentage').textContent = progressPercent + '%';
        document.getElementById('current-step').textContent = jobStatus.current_step;

        // Update steps
        this.updateProcessingSteps(jobStatus.steps_completed);

        // Handle completion
        if (jobStatus.status === 'completed') {
            this.handleProcessingComplete(jobStatus);
        } else if (jobStatus.status === 'failed') {
            this.handleProcessingFailed(jobStatus);
        }
    }

    updateProcessingSteps(completedSteps) {
        document.querySelectorAll('.step').forEach((step, index) => {
            const stepNumber = index + 1;

            if (stepNumber < completedSteps) {
                step.classList.add('completed');
                step.classList.remove('active');
            } else if (stepNumber === completedSteps) {
                step.classList.add('active');
                step.classList.remove('completed');
            } else {
                step.classList.remove('active', 'completed');
            }
        });
    }

    async handleProcessingComplete(jobStatus) {
        console.log('Processing complete!');
        this.isProcessing = false;

        // Update status badge
        document.getElementById('status-badge').textContent = 'Completed';
        document.getElementById('status-badge').classList.add('completed');

        // Show success notification
        this.showNotification('Full analysis completed! Check the results panel.', 'success');

        try {
            // Fetch results
            const results = await this.fetchJobResults(this.currentJob);
            console.log('Job results:', results);

            // Display results (but keep video visible!)
            this.displayResults(results);

            // Display spectrogram if available
            if (results.spectrogram) {
                await this.displaySpectrogram(results);
            }

            // Hide processing status after completion
            setTimeout(() => {
                document.getElementById('processing-status').style.display = 'none';
            }, 2000);

        } catch (error) {
            console.error('Failed to fetch results:', error);
            this.showNotification('Failed to load results: ' + error.message, 'error');
        }
    }

    handleProcessingFailed(jobStatus) {
        console.error('Processing failed:', jobStatus.error_message);
        this.isProcessing = false;

        // Update status badge
        document.getElementById('status-badge').textContent = 'Failed';
        document.getElementById('status-badge').classList.add('failed');

        // Show error notification
        this.showNotification('Processing failed: ' + jobStatus.error_message, 'error');

        // Show reset option
        setTimeout(() => {
            this.resetToUpload();
        }, 5000);
    }

    async fetchJobResults(jobId) {
        const response = await fetch(`/jobs/${jobId}/results`);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to fetch results');
        }

        return await response.json();
    }

    displayResults(results) {
    console.log('Displaying results:', results);

    // Handle audio-only results (no summary structure)
    if (!results.summary && results.audio_detections) {
        console.log('Handling audio-only results');

        // Update counts for audio-only
        document.getElementById('object-count').textContent = 0;
        document.getElementById('audio-count').textContent = results.audio_detections.length;
        document.getElementById('total-count').textContent = results.audio_detections.length;

        // Display audio tags
        const audioTags = results.audio_detections.map(det => det.class_name || det.label).filter(Boolean);
        this.displayTags(audioTags);

        // No timeline for audio-only
        this.displayTimeline([]);
    } else {
        // Handle full pipeline results
        document.getElementById('object-count').textContent = results.summary.object_classes ? results.summary.object_classes.length : 0;
        document.getElementById('audio-count').textContent = results.summary.audio_classes ? results.summary.audio_classes.length : 0;
        document.getElementById('total-count').textContent = results.summary.total_detections || 0;

        this.displayTags(results.summary.all_tags || []);
        this.displayTimeline(results.summary.timeline || []);
    }

    // Show detailed results section
    document.getElementById('detailed-results').style.display = 'block';
}

   async displaySpectrogram(results) {
    try {
        console.log('Loading spectrogram with data:', results.spectrogram);

        // Create spectrogram directly with data
        window.currentSpectrogram = new SpectrogramVisualizer('spectrogram-svg', results.spectrogram);
        window.currentSpectrogram.render();

            if (window.currentSpectrogram) {
                // Hide placeholder, show content
                document.getElementById('spectrogram-placeholder').style.display = 'none';
                document.getElementById('spectrogram-content').style.display = 'block';

                // Setup spectrogram controls
                this.setupSpectrogramControls();

                console.log('✅ Spectrogram displayed successfully');
            }

        } catch (error) {
            console.error('Failed to display spectrogram:', error);
            this.showNotification('Spectrogram visualization failed', 'warning');
        }
    }

    setupSpectrogramControls() {
        // dB Range control
        const dbRangeSlider = document.getElementById('db-range-slider');
        const dbRangeValue = document.getElementById('db-range-value');

        if (dbRangeSlider && window.currentSpectrogram) {
            dbRangeSlider.addEventListener('input', (e) => {
                const maxDb = parseInt(e.target.value);
                const minDb = maxDb - 80; // 80 dB range

                window.currentSpectrogram.updateDbRange(minDb, maxDb);
                dbRangeValue.textContent = `${maxDb} dB`;
            });
        }
    }

    // Update video time on spectrogram
    onVideoTimeUpdate() {
        const videoPlayer = document.getElementById('video-player');
        const currentTime = videoPlayer.currentTime;

        // Update spectrogram playhead
        if (window.currentSpectrogram) {
            window.currentSpectrogram.updatePlayhead(currentTime);
        }

        // Highlight current timeline events based on video time
        document.querySelectorAll('.timeline-event').forEach(event => {
            const eventTime = parseFloat(event.dataset.timestamp);
            if (Math.abs(currentTime - eventTime) < 2) { // Within 2 seconds
                event.classList.add('current');
            } else {
                event.classList.remove('current');
            }
        });
    }

    displayTags(tags) {
        const container = document.getElementById('tags-container');

        if (tags.length === 0) {
            container.innerHTML = `
                <div class="tags-placeholder">
                    <i class="fas fa-info-circle"></i>
                    <p>No war-related content detected</p>
                </div>
            `;
            return;
        }

        const tagsList = tags.map(tag => {
            const tagClass = this.getTagClass(tag);
            return `<span class="tag ${tagClass}">${tag.replace('_', ' ')}</span>`;
        }).join('');

        container.innerHTML = `<div class="tags-list">${tagsList}</div>`;
    }

    displayTimeline(timeline) {
        const container = document.getElementById('timeline-container');

        if (timeline.length === 0) {
            container.innerHTML = `
                <div class="timeline-placeholder">
                    <i class="fas fa-info-circle"></i>
                    <p>No detection events to display</p>
                </div>
            `;
            return;
        }

        const events = timeline.map(event => {
            return `
                <div class="timeline-event" data-timestamp="${event.timestamp}">
                    <span class="event-time">${this.formatTimestamp(event.timestamp)}</span>
                    <div class="event-type ${event.type}"></div>
                    <div class="event-info">
                        <div class="event-class">${event.class.replace('_', ' ')}</div>
                        <div class="event-confidence">Confidence: ${(event.confidence * 100).toFixed(1)}%</div>
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = `<div class="timeline-events">${events}</div>`;
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab panels
        document.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');
    }

    seekToTimestamp(timestamp) {
        const videoPlayer = document.getElementById('video-player');
        if (videoPlayer && !isNaN(timestamp)) {
            videoPlayer.currentTime = timestamp;
        }
    }

    onVideoTimeUpdate() {
        // Highlight current timeline events based on video time
        const videoPlayer = document.getElementById('video-player');
        const currentTime = videoPlayer.currentTime;

        document.querySelectorAll('.timeline-event').forEach(event => {
            const eventTime = parseFloat(event.dataset.timestamp);
            if (Math.abs(currentTime - eventTime) < 2) { // Within 2 seconds
                event.classList.add('current');
            } else {
                event.classList.remove('current');
            }
        });
    }

    showNotification(message, type = 'info') {
        const container = document.getElementById('notification-container');
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;

        notification.innerHTML = `
            ${message}
            <button class="notification-close">&times;</button>
        `;

        // Add close functionality
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.remove();
        });

        container.appendChild(notification);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    resetToUpload() {
        // Reset state
        this.currentJob = null;
        this.currentFile = null;
        this.isProcessing = false;

        // Hide all sections
        this.hideAllSections();

        // Show upload section
        document.getElementById('upload-section').style.display = 'block';
        document.getElementById('upload-progress').style.display = 'none';

        // Reset file input
        document.getElementById('file-input').value = '';
    }

    // Utility methods
    getTagClass(tag) {
        const objectClasses = ['tank', 'helicopter', 'weapon'];
        const audioClasses = ['gunshot_gunfire', 'explosion'];

        if (objectClasses.includes(tag)) return 'object';
        if (audioClasses.includes(tag)) return 'audio';
        return 'fused';
    }

    formatTimestamp(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    formatDuration(seconds) {
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);

        if (hours > 0) {
            return `${hours}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        } else {
            return `${mins}:${secs.toString().padStart(2, '0')}`;
        }
    }

    formatFileSize(bytes) {
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        if (bytes === 0) return '0 Bytes';
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    // Real-time detection methods
    startRealTimeDetection() {
        if (this.realTimeDetectionActive) return;

        console.log('Starting real-time detection');
        this.realTimeDetectionActive = true;

        // Process frames every 500ms (2 FPS for detection)
        this.detectionInterval = setInterval(() => {
            this.processCurrentFrame();
        }, 500);
    }

    stopRealTimeDetection() {
        console.log('Stopping real-time detection');
        this.realTimeDetectionActive = false;

        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }

        // Clear overlays
        this.clearDetectionOverlays();
    }

    async processCurrentFrame() {
        const videoPlayer = document.getElementById('video-player');
        if (!videoPlayer || videoPlayer.paused || videoPlayer.ended) return;

        try {
            // Create canvas to capture current frame
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            canvas.width = videoPlayer.videoWidth;
            canvas.height = videoPlayer.videoHeight;

            // Draw current video frame to canvas
            ctx.drawImage(videoPlayer, 0, 0, canvas.width, canvas.height);

            // Convert canvas to blob
            canvas.toBlob(async (blob) => {
                if (!blob || !this.realTimeDetectionActive) return;

                // Send frame to backend for detection
                const formData = new FormData();
                formData.append('file', blob, 'frame.jpg');

                try {
                    const response = await fetch('/detect/frame', {
                        method: 'POST',
                        body: formData
                    });

                    if (response.ok) {
                        const result = await response.json();
                        this.updateDetectionOverlays(result.detections, result.frame_info);
                    }
                } catch (error) {
                    console.error('Real-time detection error:', error);
                }
            }, 'image/jpeg', 0.8);

        } catch (error) {
            console.error('Frame capture error:', error);
        }
    }

    updateDetectionOverlays(detections, frameInfo) {
    const videoPlayer = document.getElementById('video-player');

    // Store detections with timestamp
    if (detections.length > 0) {
        const currentTime = videoPlayer.currentTime;
        detections.forEach(detection => {
            this.realTimeDetections.push({
                timestamp: currentTime,
                class: detection.class,
                confidence: detection.confidence,
                type: 'object'
            });
        });

        // Update detection counts and display
        const uniqueClasses = [...new Set(this.realTimeDetections.map(d => d.class))];
        document.getElementById('object-count').textContent = uniqueClasses.length;
        document.getElementById('total-count').textContent = uniqueClasses.length;

        // Update tags display
        this.displayTags(uniqueClasses);

        // Create timeline events
        const timelineEvents = this.realTimeDetections.map(detection => ({
            timestamp: detection.timestamp,
            class: detection.class,
            confidence: detection.confidence,
            type: 'object'
        }));
        this.displayTimeline(timelineEvents);
    }

    const videoContainer = document.getElementById('video-container');
    const overlay = document.getElementById('detection-overlay');

    if (!overlay || !videoPlayer) return;

    // Clear existing overlays
    overlay.innerHTML = '';

    // Calculate scaling factors
    const scaleX = videoPlayer.offsetWidth / frameInfo.width;
    const scaleY = videoPlayer.offsetHeight / frameInfo.height;

    // Draw detection boxes
    detections.forEach(detection => {
        const [x1, y1, x2, y2] = detection.bbox;

        // Scale coordinates to video display size
        const scaledX1 = x1 * scaleX;
        const scaledY1 = y1 * scaleY;
        const scaledX2 = x2 * scaleX;
        const scaledY2 = y2 * scaleY;

        // Create bounding box element
        const box = document.createElement('div');
        box.className = 'detection-box';
        box.style.left = scaledX1 + 'px';
        box.style.top = scaledY1 + 'px';
        box.style.width = (scaledX2 - scaledX1) + 'px';
        box.style.height = (scaledY2 - scaledY1) + 'px';
        box.style.borderColor = `rgb(${detection.color.join(',')})`;
        box.style.backgroundColor = `rgba(${detection.color.join(',')}, 0.2)`;

        // Create label
        const label = document.createElement('div');
        label.className = 'detection-label';
        label.textContent = `${detection.class}: ${(detection.confidence * 100).toFixed(0)}%`;
        label.style.backgroundColor = `rgb(${detection.color.join(',')})`;

        box.appendChild(label);
        overlay.appendChild(box);
    });
}

    clearDetectionOverlays() {
        const overlay = document.getElementById('detection-overlay');
        if (overlay) {
            overlay.innerHTML = '';
        }
    }

    // Image-specific processing
    async processImageInstantly(uploadResult) {
        console.log('Processing image instantly...');

        try {
            // Create form data with the image file
            const response = await fetch(`/image/${uploadResult.filename}`);
            const blob = await response.blob();

            const formData = new FormData();
            formData.append('file', blob, 'image.jpg');

            // Send to real-time detection endpoint
            const detectionResponse = await fetch('/detect/frame', {
                method: 'POST',
                body: formData
            });

            if (detectionResponse.ok) {
                const result = await detectionResponse.json();
                console.log('Image detection result:', result);

                // Show overlays on image
                this.updateImageDetectionOverlays(result.detections, result.frame_info);

                // Show notification
                if (result.detections.length > 0) {
                    this.showNotification(`Detected ${result.detections.length} objects in image!`, 'success');
                } else {
                    this.showNotification('No war-related objects detected in image', 'info');
                }
            }

        } catch (error) {
            console.error('Image detection failed:', error);
            this.showNotification('Image detection failed: ' + error.message, 'error');
        }
    }

    // Video-specific processing (skip deduplication)
    async startVideoProcessing(filename) {
        console.log('Starting fast video processing (audio + object, skip deduplication)...');

        // Start fast background processing
        try {
            const response = await fetch(`/process_fast/${filename}`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error('Fast processing failed to start');
            }

            const result = await response.json();
            this.currentJob = result.job_id;
            this.isProcessing = true;

            // Subscribe to job updates
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({
                    type: 'subscribe_job',
                    job_id: this.currentJob
                }));
            }

            // DON'T show processing status - let it run silently in background
            // this.showProcessingStatus();

            // Just show a small notification that background processing started
            this.showNotification('Background analysis started. Video ready to play!', 'info');

        } catch (error) {
            console.error('Fast video processing start failed:', error);
            this.showNotification('Background processing failed to start: ' + error.message, 'error');
        }
    }

    // Image overlay rendering
    updateImageDetectionOverlays(detections, frameInfo) {
        const imageContainer = document.getElementById('image-container');
        const imageDisplay = document.getElementById('image-display');
        const overlay = document.getElementById('image-detection-overlay');

        if (!overlay || !imageDisplay) return;

        // Clear existing overlays
        overlay.innerHTML = '';

        // Wait for image to load to get proper dimensions
        if (imageDisplay.complete) {
            this.renderImageOverlays(detections, frameInfo, imageDisplay, overlay);
        } else {
            imageDisplay.onload = () => {
                this.renderImageOverlays(detections, frameInfo, imageDisplay, overlay);
            };
        }
    }

    renderImageOverlays(detections, frameInfo, imageDisplay, overlay) {
        // Get image display dimensions
        const displayWidth = imageDisplay.offsetWidth;
        const displayHeight = imageDisplay.offsetHeight;

        // Calculate scaling factors
        const scaleX = displayWidth / frameInfo.width;
        const scaleY = displayHeight / frameInfo.height;

        // Draw detection boxes
        detections.forEach(detection => {
            const [x1, y1, x2, y2] = detection.bbox;

            // Scale coordinates to image display size
            const scaledX1 = x1 * scaleX;
            const scaledY1 = y1 * scaleY;
            const scaledX2 = x2 * scaleX;
            const scaledY2 = y2 * scaleY;

            // Create bounding box element
            const box = document.createElement('div');
            box.className = 'detection-box';
            box.style.left = scaledX1 + 'px';
            box.style.top = scaledY1 + 'px';
            box.style.width = (scaledX2 - scaledX1) + 'px';
            box.style.height = (scaledY2 - scaledY1) + 'px';
            box.style.borderColor = `rgb(${detection.color.join(',')})`;
            box.style.backgroundColor = `rgba(${detection.color.join(',')}, 0.2)`;

            // Create label
            const label = document.createElement('div');
            label.className = 'detection-label';
            label.textContent = `${detection.class}: ${(detection.confidence * 100).toFixed(0)}%`;
            label.style.backgroundColor = `rgb(${detection.color.join(',')})`;

            box.appendChild(label);
            overlay.appendChild(box);
        });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new WarDetectionApp();
});