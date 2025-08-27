/**
 * War Detection Pipeline - D3.js Spectrogram Visualizer
 * Professional mel spectrogram visualization for audio analysis
 */

class SpectrogramVisualizer {
    constructor(containerId, spectrogramData) {
        this.containerId = containerId;
        this.data = spectrogramData;
        this.svg = null;
        this.colorScale = null;
        this.xScale = null;
        this.yScale = null;

        // Configuration
        this.config = {
            width: 450,
            height: 180,
            margin: { top: 10, right: 10, bottom: 30, left: 40 },
            colorScheme: 'viridis', // Professional color scheme
            dbRange: [-80, 0], // dB range for visualization
            animationDuration: 300
        };

        // Calculate inner dimensions
        this.innerWidth = this.config.width - this.config.margin.left - this.config.margin.right;
        this.innerHeight = this.config.height - this.config.margin.top - this.config.margin.bottom;

        console.log('SpectrogramVisualizer initialized with data:', this.data);
    }

    render() {
        console.log('Rendering spectrogram...');

        if (!this.data || !this.data.data) {
            console.error('No spectrogram data available');
            return;
        }

        // Clear existing content
        d3.select(`#${this.containerId}`).selectAll("*").remove();

        // Create SVG
        this.svg = d3.select(`#${this.containerId}`)
            .attr('width', this.config.width)
            .attr('height', this.config.height);

        const g = this.svg.append('g')
            .attr('transform', `translate(${this.config.margin.left},${this.config.margin.top})`);

        // Setup scales
        this.setupScales();

        // Create heatmap
        this.createHeatmap(g);

        // Add axes
        this.addAxes(g);

        // Add interactive features
        this.addInteractivity(g);

        console.log('✅ Spectrogram rendered successfully');
    }

    setupScales() {
        const spectrogramMatrix = this.data.data;
        const numTimeFrames = spectrogramMatrix[0].length;
        const numFreqBins = spectrogramMatrix.length;

        // Time scale (X-axis) - use actual duration if available, otherwise calculate from frames
        const actualDuration = this.data.duration || (numTimeFrames * 0.032); // assume ~32ms per frame
        this.xScale = d3.scaleLinear()
            .domain([0, actualDuration])
            .range([0, this.innerWidth]);

        // Frequency scale (Y-axis) - inverted for typical spectrogram display
        this.yScale = d3.scaleLinear()
            .domain([0, numFreqBins])
            .range([this.innerHeight, 0]);

        // Color scale for magnitude (dB values)
        const [minDb, maxDb] = this.config.dbRange;
        this.colorScale = d3.scaleSequential()
            .domain([minDb, maxDb])
            .interpolator(this.getColorInterpolator());

        console.log(`Scales setup: ${numTimeFrames} time frames, ${numFreqBins} frequency bins`);
    }

    getColorInterpolator() {
        // Professional color schemes for spectrograms
        const schemes = {
            'viridis': d3.interpolateViridis,
            'plasma': d3.interpolatePlasma,
            'inferno': d3.interpolateInferno,
            'magma': d3.interpolateMagma,
            'cool': d3.interpolateCool,
            'warm': d3.interpolateWarm
        };

        return schemes[this.config.colorScheme] || d3.interpolateViridis;
    }

    createHeatmap(g) {
    const spectrogramMatrix = this.data.data;
    const numTimeFrames = spectrogramMatrix[0].length;
    const numFreqBins = spectrogramMatrix.length;

    // Calculate cell dimensions
    const cellWidth = this.innerWidth / numTimeFrames;
    const cellHeight = this.innerHeight / numFreqBins;

    // Create data array for D3
    const heatmapData = [];
    for (let freqBin = 0; freqBin < numFreqBins; freqBin++) {
        for (let timeFrame = 0; timeFrame < numTimeFrames; timeFrame++) {
            const magnitude = spectrogramMatrix[freqBin][timeFrame];
            heatmapData.push({
                timeFrame: timeFrame,
                freqBin: freqBin,
                magnitude: magnitude,
                time: (timeFrame / numTimeFrames) * (this.data.duration || numTimeFrames),
                frequency: freqBin
            });
        }
    }

    // Create heatmap cells
    const cells = g.selectAll('.spectrogram-cell')
        .data(heatmapData)
        .enter()
        .append('rect')
        .attr('class', 'spectrogram-cell')
        .attr('x', d => this.xScale(d.time))
        .attr('y', d => this.yScale(d.freqBin + 1))
        .attr('width', cellWidth)
        .attr('height', cellHeight)
        .attr('fill', d => this.colorScale(d.magnitude))
        .attr('opacity', 0)
        .on('mouseover', (event, d) => this.showTooltip(event, d))
        .on('mouseout', () => this.hideTooltip());

    // Animate cells appearing
    cells.transition()
        .duration(this.config.animationDuration)
        .delay((d, i) => i * 0.5)
        .attr('opacity', 0.9);

    console.log(`Created ${heatmapData.length} spectrogram cells`);
}

    addAxes(g) {
        // X-axis (Time)
        const xAxis = d3.axisBottom(this.xScale)
            .tickFormat(d => {
                const minutes = Math.floor(d / 60);
                const seconds = Math.floor(d % 60);
                return `${minutes}:${seconds.toString().padStart(2, '0')}`;
            })
            .ticks(6);

        g.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0,${this.innerHeight})`)
            .call(xAxis)
            .append('text')
            .attr('x', this.innerWidth / 2)
            .attr('y', 25)
            .attr('fill', 'white')
            .attr('font-size', '12px')
            .attr('text-anchor', 'middle')
            .text('Time');

        // Y-axis (Frequency)
        const yAxis = d3.axisLeft(this.yScale)
            .tickFormat(d => `${Math.round(d)}`)
            .ticks(5);

        g.append('g')
            .attr('class', 'y-axis')
            .call(yAxis)
            .append('text')
            .attr('transform', 'rotate(-90)')
            .attr('x', -this.innerHeight / 2)
            .attr('y', -25)
            .attr('fill', 'white')
            .attr('font-size', '12px')
            .attr('text-anchor', 'middle')
            .text('Mel Frequency Bins');

        // Style axes
        g.selectAll('.x-axis, .y-axis')
            .selectAll('path, line')
            .attr('stroke', 'rgba(255, 255, 255, 0.3)');

        g.selectAll('.x-axis, .y-axis')
            .selectAll('text')
            .attr('fill', 'rgba(255, 255, 255, 0.8)')
            .attr('font-size', '11px');
    }

    addInteractivity(g) {
        // Add click-to-seek functionality
        const overlay = g.append('rect')
            .attr('class', 'spectrogram-overlay')
            .attr('width', this.innerWidth)
            .attr('height', this.innerHeight)
            .attr('fill', 'transparent')
            .attr('cursor', 'pointer')
            .on('click', (event) => {
                const [mouseX] = d3.pointer(event);
                const clickedTime = this.xScale.invert(mouseX);
                this.seekToTime(clickedTime);
            });

        // Add playhead indicator
        this.playhead = g.append('line')
            .attr('class', 'playhead')
            .attr('x1', 0)
            .attr('x2', 0)
            .attr('y1', 0)
            .attr('y2', this.innerHeight)
            .attr('stroke', '#FF4444')
            .attr('stroke-width', 2)
            .attr('opacity', 0);
    }

    updatePlayhead(currentTime) {
        if (this.playhead && this.xScale) {
            const x = this.xScale(currentTime);
            this.playhead
                .attr('x1', x)
                .attr('x2', x)
                .attr('opacity', 1);
        }
    }

    addDetectionMarkers(audioDetections) {
        if (!audioDetections || audioDetections.length === 0) return;

        const g = this.svg.select('g');

        // Add detection markers
        const markers = g.selectAll('.detection-marker')
            .data(audioDetections)
            .enter()
            .append('g')
            .attr('class', 'detection-marker');

        // Add detection rectangles
        markers.append('rect')
            .attr('x', d => this.xScale(d.chunk_start || d.timestamp))
            .attr('y', 0)
            .attr('width', d => Math.max(2, this.xScale((d.chunk_end || d.timestamp + 1) - (d.chunk_start || d.timestamp))))
            .attr('height', this.innerHeight)
            .attr('fill', d => this.getDetectionColor(d.class))
            .attr('opacity', 0.3)
            .attr('stroke', d => this.getDetectionColor(d.class))
            .attr('stroke-width', 1);

        // Add detection labels
        markers.append('text')
            .attr('x', d => this.xScale(d.chunk_start || d.timestamp) + 2)
            .attr('y', 15)
            .attr('fill', 'white')
            .attr('font-size', '10px')
            .attr('font-weight', 'bold')
            .text(d => d.class.replace('_', ' '));

        console.log(`Added ${audioDetections.length} detection markers to spectrogram`);
    }

    getDetectionColor(className) {
        const colors = {
            'gunshot_gunfire': '#FF4444',
            'explosion': '#FF8800',
            'helicopter': '#44FF44'
        };
        return colors[className] || '#FFFFFF';
    }

    showTooltip(event, d) {
        // Create or update tooltip
        let tooltip = d3.select('.spectrogram-tooltip');

        if (tooltip.empty()) {
            tooltip = d3.select('body')
                .append('div')
                .attr('class', 'spectrogram-tooltip')
                .style('position', 'absolute')
                .style('background', 'rgba(0, 0, 0, 0.8)')
                .style('color', 'white')
                .style('padding', '8px')
                .style('border-radius', '4px')
                .style('font-size', '12px')
                .style('pointer-events', 'none')
                .style('z-index', '1000')
                .style('opacity', 0);
        }

        tooltip.html(`
            <div>Time: ${d.time.toFixed(2)}s</div>
            <div>Frequency Bin: ${d.freqBin}</div>
            <div>Magnitude: ${d.magnitude.toFixed(1)} dB</div>
        `)
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px')
        .transition()
        .duration(200)
        .style('opacity', 1);
    }

    hideTooltip() {
        d3.select('.spectrogram-tooltip')
            .transition()
            .duration(200)
            .style('opacity', 0);
    }

    seekToTime(time) {
        // Notify parent application to seek video
        const event = new CustomEvent('spectrogramSeek', {
            detail: { time: time }
        });
        document.dispatchEvent(event);

        console.log(`Spectrogram seek to: ${time.toFixed(2)}s`);
    }

    updateColorScheme(scheme) {
        this.config.colorScheme = scheme;
        this.colorScale = d3.scaleSequential()
            .domain(this.config.dbRange)
            .interpolator(this.getColorInterpolator());

        // Update existing cells
        this.svg.selectAll('.spectrogram-cell')
            .transition()
            .duration(500)
            .attr('fill', d => this.colorScale(d.magnitude));
    }

    updateDbRange(minDb, maxDb) {
        this.config.dbRange = [minDb, maxDb];
        this.colorScale = d3.scaleSequential()
            .domain([minDb, maxDb])
            .interpolator(this.getColorInterpolator());

        // Update existing cells
        this.svg.selectAll('.spectrogram-cell')
            .transition()
            .duration(300)
            .attr('fill', d => this.colorScale(d.magnitude));

        console.log(`Updated dB range: ${minDb} to ${maxDb}`);
    }

    resize(width, height) {
        this.config.width = width;
        this.config.height = height;
        this.innerWidth = width - this.config.margin.left - this.config.margin.right;
        this.innerHeight = height - this.config.margin.top - this.config.margin.bottom;

        // Re-render with new dimensions
        this.render();
    }

    // Static method to create spectrogram from API data
    static async createFromJobId(jobId, containerId) {
        try {
            console.log(`Fetching spectrogram data for job: ${jobId}`);

            const response = await fetch(`/jobs/${jobId}/spectrogram`);
            if (!response.ok) {
                throw new Error('Failed to fetch spectrogram data');
            }

            const spectrogramData = await response.json();
            console.log('Spectrogram data received:', spectrogramData);

            const visualizer = new SpectrogramVisualizer(containerId, spectrogramData);
            visualizer.render();

            return visualizer;

        } catch (error) {
            console.error('Failed to create spectrogram:', error);
            return null;
        }
    }
}

// Global spectrogram instance
window.SpectrogramVisualizer = SpectrogramVisualizer;
window.currentSpectrogram = null;

// Listen for spectrogram seek events
document.addEventListener('spectrogramSeek', (event) => {
    const time = event.detail.time;
    const videoPlayer = document.getElementById('video-player');

    if (videoPlayer) {
        videoPlayer.currentTime = time;
        console.log(`Video seeked to: ${time.toFixed(2)}s from spectrogram click`);
    }
});

// Export for use in main.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SpectrogramVisualizer;
}