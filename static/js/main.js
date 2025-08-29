/**
 * Clean War Detection Pipeline - Frontend Application
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
        this.realTimeDetections = [];

        this.init();
    }

    init() {
        console.log('Initializing War Detection Pipeline App');

        this.setupEventListeners();
        this.initWebSocket();
        this.setupDragAndDrop();
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

        // Video player events
        const videoPlayer = document.getElementById('video-player');
        if (videoPlayer) {
            videoPlayer.addEventListener('timeupdate', () => {
                this.onVideoTimeUpdate();
            });

            videoPlayer.addEventListener('play', () => {
                this.startRealTimeDetection();
            });

            videoPlayer.addEventListener('pause', () => {
                this.stopRealTimeDetection();
            });

            videoPlayer.addEventListener('seeked', () => {
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
        this.hideAllSections();
        document.getElementById('upload-section').style.display = 'block';
        this.updateConnectionStatus('connecting');
    }

    hideAllSections() {
        const mediaDisplay = document.getElementById('media-display');
        const processingSta

tus = document.getElementById('processing-status');
        const detailedResults = document.getElementById('detailed-results');

        if (mediaDisplay) mediaDisplay.style.display = 'none';
        if (processingStatus) processingStatus.style.display = 'none';
        if (detailedResults) detailedResults.style.display = 'none';
    }

    initWebSocket() {
        console.log('Connecting to WebSocket...');

        try {
            this.websocket = new WebSocket(this.config.wsUrl);

            this.websocket.onopen = () => {
                console.log('✅ WebSocket connected');
                this.updateConnectionStatus('connected');
            };

            this.websocket.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleWebSocketMessage(message);
            };

            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus('disconnected');
                setTimeout(() => this.initWebSocket(), 3000);
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
        if (!statusElement) return;

        const statusMap = {
            'connecting': { text: 'Connecting...', class: 'connecting' },
            'connected': { text: 'Connected', class: 'connected' },
            'disconnected': { text: 'Disconnected', class: 'disconnected' }
        };

        const statusInfo = statusMap[status] || statusMap['disconnected'];
        statusElement.innerHTML = statusInfo.text;
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

        if (!this.validateFile(file)) return;

        this.currentFile = file;
        this.showUploadProgress();

        try {
            const uploadResult = await this.uploadFile(file);
            console.log('Upload result:', uploadResult);

            this.displayMedia(uploadResult);

            if (uploadResult.file_type === 'image') {
                await this.processImageInstantly(uploadResult);
            } else if (uploadResult.file_type === 'video') {
                this.showNotification('Video ready! Press play for real-time detection.', 'info');
                await this.startAudioAnalysisOnly(uploadResult.filename);
            }

        } catch (error) {
            console.error('File handling failed:', error);
            this.showNotification('Upload failed: ' + error.message, 'error');
            this.resetToUpload();
        }
    }

    validateFile(file) {
        if (file.size > this.config.maxFileSize) {
            const maxSizeMB = (this.config.maxFileSize / (1024 * 1024)).toFixed(1);
            this.showNotification(`File too large. Maximum size: ${maxSizeMB}MB`, 'error');
            return false;
        }

        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        if (!this.config.allowedFormats.includes(fileExtension)) {
            this.showNotification(`File type not supported. Allowed: ${this.config.allowedFormats.join(', ')}`, 'error');
            return false;
        }

        return true;
    }

    showUploadProgress() {
        const uploadProgress = document.getElementById('upload-progress');
        const progressFill = document.getElementById('upload-progress-fill');
        const progressText = document.getElementById('upload-progress-text');

        if (!uploadProgress) return;

        uploadProgress.style.display = 'block';
        progressFill.style.width = '0%';
        progressText.textContent = 'Uploading...';

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

        if (progressFill && progressText) {
            progressFill.style.width = '100%';
            progressText.textContent = 'Upload complete!';
        }

        return await response.json();
    }

    displayMedia(uploadResult) {
        const uploadSection = document.getElementById('upload-section');
        const uploadProgress = document.getElementById('upload-progress');
        const mediaDisplay = document.getElementById('media-display');

        if (uploadSection) uploadSection.style.display = 'none';
        if (uploadProgress) uploadProgress.style.display = 'none';
        if (mediaDisplay) mediaDisplay.style.display = 'block';

        // Update media info
        const mediaTitle = document.getElementById('media-title');
        const fileSize = document.getElementById('file-size');
        const fileType = document.getElementById('file-type');

        if (mediaTitle) mediaTitle.textContent = uploadResult.original_filename;
        if (fileSize) fileSize.textContent = `Size: ${this.formatFileSize(uploadResult.file_size)}`;
        if (fileType) fileType.textContent = `Type: ${uploadResult.file_type.toUpperCase()}`;

        if (uploadResult.file_type === 'video') {
            this.displayVideo(uploadResult);
        } else {
            this.displayImage(uploadResult);
        }
    }

    displayVideo(uploadResult) {
        const videoContainer = document.getElementById('video-container');
        const imageContainer = document.getElementById('image-container');
        const videoPlayer = document.getElementById('video-player');

        if (videoContainer) videoContainer.style.display = 'block';
        if (imageContainer) imageContainer.style.display = 'none';

        if (videoPlayer) {
            videoPlayer.src = `/video/${uploadResult.filename}`;
            videoPlayer.load();

            videoPlayer.addEventListener('loadedmetadata', () => {
                const fileDuration = document.getElementById('file-duration');
                if (fileDuration) {
                    const duration = this.formatDuration(videoPlayer.duration);
                    fileDuration.textContent = `Duration: ${duration}`;
                }
            });
        }
    }

    displayImage(uploadResult) {
        const videoContainer = document.getElementById('video-container');
        const imageContainer = document.getElementById('image-container');
        const imageDisplay = document.getElementById('image-display');
        const fileDuration = document.getElementById('file-duration');

        if (videoContainer) videoContainer.style.display = 'none';
        if (imageContainer) imageContainer.style.display = 'block';
        if (fileDuration) fileDuration.textContent = 'Duration: --';

        if (imageDisplay) {
            imageDisplay.src = `/image/${uploadResult.filename}`;
        }
    }

    async startAudioAnalysisOnly(filename) {
        console.log('Starting audio-only analysis in background...');

        try {
            const response = await fetch(`/process_audio_only/${filename}`, {
                method: 'POST'
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Audio analysis response error:', errorText);
                throw new Error(`Audio analysis failed: ${response.status}`);
            }

            const result = await response.json();
            console.log('Audio analysis started:', result);

            this.currentJob = result.job_id;

            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({
                    type: 'subscribe_job',
                    job_id: this.currentJob
                }));
            }

            this.showNotification('Audio analysis started in background...', 'info');

        } catch (error) {
            console.error('Audio analysis start failed:', error);
            this.showNotification('Audio analysis failed to start: ' + error.message, 'warning');
        }
    }

    handleJobUpdate(jobStatus) {
        console.log('Job update:', jobStatus);

        if (jobStatus.job_id !== this.currentJob) return;

        const progressFill = document.getElementById('processing-progress-fill');
        const progressPercentage = document.getElementById('progress-percentage');
        const currentStep = document.getElementById('current-step');

        if (progressFill) {
            const progressPercent = Math.round(jobStatus.progress * 100);
            progressFill.style.width = progressPercent + '%';
            if (progressPercentage) progressPercentage.textContent = progressPercent + '%';
            if (currentStep) currentStep.textContent = jobStatus.current_step || 'Processing...';
        }

        this.updateProcessingSteps(jobStatus.steps_completed);

        if (jobStatus.status === 'completed') {
            this.handleProcessingComplete(jobStatus);
        } else if (jobStatus.status === 'failed') {
            this.handleProcessingFailed(jobStatus);
        }
    }

    updateProcessingSteps(completedSteps) {
        document.querySelectorAll('.step').forEach((step, index) => {
            const stepNumber = index + 1;

            step.classList.remove('active', 'completed');

            if (stepNumber < completedSteps) {
                step.classList.add('completed');
            } else if (stepNumber === completedSteps) {
                step.classList.add('active');
            }
        });
    }

    async handleProcessingComplete(jobStatus) {
        console.log('Processing complete!');
        this.isProcessing = false;

        const statusBadge = document.getElementById('status-badge');
        if (statusBadge) {
            statusBadge.textContent = 'Completed';
            statusBadge.classList.add('completed');
        }

        this.showNotification('Analysis completed! Check the results.', 'success');

        try {
            const results = await this.fetchJobResults(this.currentJob);
            console.log('Job results:', results);

            this.displayResults(results);

            if (results.spectrogram) {
                await this.displaySpectrogram(results);
            }

            setTimeout(() => {
                const processingStatus = document.getElementById('processing-status');
                if (processingStatus) processingStatus.style.display = 'none';
            }, 2000);

        } catch (error) {
            console.error('Failed to fetch results:', error);
            this.showNotification('Failed to load results: ' + error.message, 'error');
        }
    }

    handleProcessingFailed(jobStatus) {
        console.error('Processing failed:', jobStatus.error_message);
        this.isProcessing = false;

        const statusBadge = document.getElementById('status-badge');
        if (statusBadge) {
            statusBadge.textContent = 'Failed';
            statusBadge.classList.add('failed');
        }

        this.showNotification('Processing failed: ' + jobStatus.error_message, 'error');

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

        const objectCount = document.getElementById('object-count');
        const audioCount = document.getElementById('audio-count');
        const totalCount = document.getElementById('total-count');

        if (!results.summary && results.audio_detections) {
            // Handle audio-only results
            if (objectCount) objectCount.textContent = '0';
            if (audioCount) audioCount.textContent = results.audio_detections.length;
            if (totalCount) totalCount.textContent = results.audio_detections.length;

            const audioTags = results.audio_detections.map(det => det.class_name || det.label).filter(Boolean);
            this.displayTags(audioTags);
            this.displayTimeline([]);
        } else {
            // Handle full pipeline results
            if (objectCount) objectCount.textContent = results.summary?.object_classes?.length || 0;
            if (audioCount) audioCount.textContent = results.summary?.audio_classes?.length || 0;
            if (totalCount) totalCount.textContent = results.summary?.total_detections || 0;

            this.displayTags(results.summary?.all_tags || []);
            this.displayTimeline(results.summary?.timeline || []);
        }
    }

    async displaySpectrogram(results) {
        try {
            console.log('Loading spectrogram with data:', results.spectrogram);

            window.currentSpectrogram = new SpectrogramVisualizer('spectrogram-svg', results.spectrogram);
            window.currentSpectrogram.render();

            if (window.currentSpectrogram) {
                const spectrogramPlaceholder = document.getElementById('spectrogram-placeholder');
                const spectrogramContent = document.getElementById('spectrogram-content');

                if (spectrogramPlaceholder) spectrogramPlaceholder.style.display = 'none';
                if (spectrogramContent) spectrogramContent.style.display = 'block';

                this.setupSpectrogramControls();
                console.log('✅ Spectrogram displayed successfully');
            }

        } catch (error) {
            console.error('Failed to display spectrogram:', error);
            this.showNotification('Spectrogram visualization failed', 'warning');
        }
    }

    setupSpectrogramControls() {
        const dbRangeSlider = document.getElementById('db-range-slider');
        const dbRangeValue = document.getElementById('db-range-value');

        if (dbRangeSlider && window.currentSpectrogram && dbRangeValue) {
            dbRangeSlider.addEventListener('input', (e) => {
                const maxDb = parseInt(e.target.value);
                const minDb = maxDb - 80;

                window.currentSpectrogram.updateDbRange(minDb, maxDb);
                dbRangeValue.textContent = `${maxDb} dB`;
            });
        }
    }

    displayTags(tags) {
        const container = document.getElementById('tags-container');
        if (!container) return;

        if (tags.length === 0) {
            container.innerHTML = `
                <div class="tags-placeholder">
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
        if (!container) return;

        if (timeline.length === 0) {
            container.innerHTML = `
                <div class="timeline-placeholder">
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

    onVideoTimeUpdate() {
        const videoPlayer = document.getElementById('video-player');
        if (!videoPlayer) return;

        const currentTime = videoPlayer.currentTime;

        // Update spectrogram playhead
        if (window.currentSpectrogram) {
            window.currentSpectrogram.updatePlayhead(currentTime);
        }

        // Highlight current timeline events
        document.querySelectorAll('.timeline-event').forEach(event => {
            const eventTime = parseFloat(event.dataset.timestamp);
            if (Math.abs(currentTime - eventTime) < 2) {
                event.classList.add('current');
            } else {
                event.classList.remove('current');
            }
        });
    }

    seekToTimestamp(timestamp) {
        const videoPlayer = document.getElementById('video-player');
        if (videoPlayer && !isNaN(timestamp)) {
            videoPlayer.currentTime = timestamp;
        }
    }

    // Real-time detection methods
    startRealTimeDetection() {
        if (this.realTimeDetectionActive) return;

        console.log('Starting real-time detection');
        this.realTimeDetectionActive = true;

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

        this.clearDetectionOverlays();
    }

    async processCurrentFrame() {
        const videoPlayer = document.getElementById('video-player');
        if (!videoPlayer || videoPlayer.paused || videoPlayer.ended) return;

        try {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            canvas.width = videoPlayer.videoWidth;
            canvas.height = videoPlayer.videoHeight;

            ctx.drawImage(videoPlayer, 0, 0, canvas.width, canvas.height);

            canvas.toBlob(async (blob) => {
                if (!blob || !this.realTimeDetectionActive) return;

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
        const overlay = document.getElementById('detection-overlay');

        if (!overlay || !videoPlayer) return;

        // Store detections for timeline
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

            // Update counts and display
            const uniqueClasses = [...new Set(this.realTimeDetections.map(d => d.class))];
            const objectCount = document.getElementById('object-count');
            const totalCount = document.getElementById('total-count');

            if (objectCount) objectCount.textContent = uniqueClasses.length;
            if (totalCount) totalCount.textContent = uniqueClasses.length;

            this.displayTags(uniqueClasses);

            const timelineEvents = this.realTimeDetections.map(detection => ({
                timestamp: detection.timestamp,
                class: detection.class,
                confidence: detection.confidence,
                type: 'object'
            }));
            this.displayTimeline(timelineEvents);
        }

        // Clear existing overlays
        overlay.innerHTML = '';

        // Calculate scaling factors
        const scaleX = videoPlayer.offsetWidth / frameInfo.width;
        const scaleY = videoPlayer.offsetHeight / frameInfo.height;

        // Draw detection boxes
        detections.forEach(detection => {
            const [x1, y1, x2, y2] = detection.bbox;

            const scaledX1 = x1 * scaleX;
            const scaledY1 = y1 * scaleY;
            const scaledX2 = x2 * scaleX;
            const scaledY2 = y2 * scaleY;

            const box = document.createElement('div');
            box.className = 'detection-box';
            box.style.left = scaledX1 + 'px';
            box.style.top = scaledY1 + 'px';
            box.style.width = (scaledX2 - scaledX1) + 'px';
            box.style.height = (scaledY2 - scaledY1) + 'px';
            box.style.borderColor = `rgb(${detection.color.join(',')})`;
            box.style.backgroundColor = `rgba(${detection.color.join(',')}, 0.2)`;

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

    async processImageInstantly(uploadResult) {
        console.log('Processing image instantly...');

        try {
            const response = await fetch(`/image/${uploadResult.filename}`);
            const blob = await response.blob();

            const formData = new FormData();
            formData.append('file', blob, 'image.jpg');

            const detectionResponse = await fetch('/detect/frame', {
                method: 'POST',
                body: formData
            });

            if (detectionResponse.ok) {
                const result = await detectionResponse.json();
                console.log('Image detection result:', result);

                this.updateImageDetectionOverlays(result.detections, result.frame_info);

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

    updateImageDetectionOverlays(detections, frameInfo) {
        const imageDisplay = document.getElementById('image-display');
        const overlay = document.getElementById('image-detection-overlay');

        if (!overlay || !imageDisplay) return;

        overlay.innerHTML = '';

        if (imageDisplay.complete) {
            this.renderImageOverlays(detections, frameInfo, imageDisplay, overlay);
        } else {
            imageDisplay.onload = () => {
                this.renderImageOverlays(detections, frameInfo, imageDisplay, overlay);
            };
        }
    }

    renderImageOverlays(detections, frameInfo, imageDisplay, overlay) {
        const displayWidth = imageDisplay.offsetWidth;
        const displayHeight = imageDisplay.offsetHeight;

        const scaleX = displayWidth / frameInfo.width;
        const scaleY = displayHeight / frameInfo.height;

        detections.forEach(detection => {
            const [x1, y1, x2, y2] = detection.bbox;

            const scaledX1 = x1 * scaleX;
            const scaledY1 = y1 * scaleY;
            const scaledX2 = x2 * scaleX;
            const scaledY2 = y2 * scaleY;

            const box = document.createElement('div');
            box.className = 'detection-box';
            box.style.left = scaledX1 + 'px';
            box.style.top = scaledY1 + 'px';
            box.style.width = (scaledX2 - scaledX1) + 'px';
            box.style.height = (scaledY2 - scaledY1) + 'px';
            box.style.borderColor = `rgb(${detection.color.join(',')})`;
            box.style.backgroundColor = `rgba(${detection.color.join(',')}, 0.2)`;

            const label = document.createElement('div');
            label.className = 'detection-label';
            label.textContent = `${detection.class}: ${(detection.confidence * 100).toFixed(0)}%`;
            label.style.backgroundColor = `rgb(${detection.color.join(',')})`;

            box.appendChild(label);
            overlay.appendChild(box);
        });
    }

    showNotification(message, type = 'info') {
        const container = document.getElementById('notification-container');
        if (!container) return;

        const notification = document.createElement('div');
        notification.className = `notification ${type}`;

        notification.innerHTML = `
            ${message}
            <button class="notification-close">&times;</button>
        `;

        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.remove();
        });

        container.appendChild(notification);

        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    resetToUpload() {
        this.currentJob = null;
        this.currentFile = null;
        this.isProcessing = false;

        this.hideAllSections();

        const uploadSection = document.getElementById('upload-section');
        const uploadProgress = document.getElementById('upload-progress');
        const fileInput = document.getElementById('file-input');

        if (uploadSection) uploadSection.style.display = 'block';
        if (uploadProgress) uploadProgress.style.display = 'none';
        if (fileInput) fileInput.value = '';
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
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new WarDetectionApp();
});