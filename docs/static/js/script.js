
// Progressive Video Loading with Intersection Observer
function initVideoLazyLoading() {
    // Create intersection observer for videos
    const videoObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const video = entry.target;
                const source = video.querySelector('source');
                
                // Add lazy-video class and loading state
                video.classList.add('lazy-video', 'loading');
                
                // Load video source
                if (video.dataset.src && !video.src) {
                    video.src = video.dataset.src;
                    if (source && source.dataset.src) {
                        source.src = source.dataset.src;
                    }
                    
                    // Load the video
                    video.load();
                    
                    // Handle video loading events
                    video.addEventListener('loadeddata', () => {
                        // Remove loading state and add loaded state
                        video.classList.remove('loading');
                        video.classList.add('loaded');
                        
                        // Start playing with smooth transition
                        setTimeout(() => {
                            video.play().catch(() => {});
                        }, 300);
                    });
                    
                    video.addEventListener('error', () => {
                        video.classList.remove('loading');
                    });
                }
                
                // Stop observing this video
                observer.unobserve(video);
            }
        });
    }, {
        // Start loading when video is 20% visible
        threshold: 0.2,
        // Start loading 150px before video comes into view
        rootMargin: '150px'
    });
    
    // Observe all videos with data-src attribute
    document.querySelectorAll('video[data-src]').forEach(video => {
        videoObserver.observe(video);
    });
}

// Progressive Image Loading with Intersection Observer
function initImageLazyLoading() {
    // Create intersection observer for images
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                
                // Load image source
                if (img.dataset.src && img.src !== img.dataset.src) {
                    // Add loading class for smooth animation
                    img.classList.add('loading');
                    
                    // Create new image to preload
                    const newImg = new Image();
                    newImg.onload = () => {
                        // Once loaded, replace src and add loaded state with smooth transition
                        img.src = img.dataset.src;
                        img.classList.remove('loading');
                        img.classList.add('loaded');
                        
                        // Trigger reflow to ensure smooth animation
                        img.offsetHeight;
                    };
                    newImg.onerror = () => {
                        // If image fails to load, show error placeholder
                        img.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 300'%3E%3Crect width='100%25' height='100%25' fill='%23f5f5f7'/%3E%3Ctext x='50%25' y='50%25' font-family='Arial' font-size='16' fill='%2386868b' text-anchor='middle' dy='.3em'%3EImage failed to load%3C/text%3E%3C/svg%3E";
                        img.classList.remove('loading');
                    };
                    newImg.src = img.dataset.src;
                }
                
                // Stop observing this image
                observer.unobserve(img);
            }
        });
    }, {
        // Start loading when image is 10% visible
        threshold: 0.1,
        // Start loading 100px before image comes into view for smoother experience
        rootMargin: '100px'
    });
    
    // Observe all images with data-src attribute
    document.querySelectorAll('img[data-src]').forEach(img => {
        imageObserver.observe(img);
    });
}


// Intersection observer for section animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.animationPlayState = 'running';
        }
    });
}, observerOptions);

document.querySelectorAll('.section').forEach(section => {
    observer.observe(section);
});

// Interactive Demo Data
const demoData = {
    datasets: [
        {
            name: "EgoDex",
            tasks: [
                { name: "Uncharge Airpods", id: "48-00000180"},
                { name: "Disassemble Furniture", id: "08-00000652" },
                { name: "Stock in Fridge", id: "14-00000033" },
                { name: "Gather Dice", id: "23-00000014" },
                { name: "Open Box", id: "60-00000559"},
                { name: "Pick and Place", id: "28-0000035c"}
            ]
        },
        {
            name: "TACO",
            tasks: [
                { name: "Scrape Off", id: "02-000000ed" },
                { name: "Pour In", id: "13-0000005e" },
                { name: "Cut Slice", id: "24-00000010" },
                { name: "Put Out", id: "36-000001f8" },
                { name: "Dust Roll", id: "40-00000052" },
                { name: "Scrape Off", id: "04-00000015" },
                { name: "Skim Off", id: "42-00000018" }
            ]
        },
        {
            name: "FPHA",
            tasks: [
                { name: "Light Candle", id: "19-00000004" },
                { name: "Pour Juice", id: "32-00000007" },
                { name: "Wash Sponge", id: "36-00000001" },
                { name: "Handshake", id: "45-00000001" }
            ]
        },
        {
            name: "H2O",
            tasks: [
                { name: "Rotate Bottle", id: "47-00000053" },
                { name: "Open Box", id: "50-00000033" },
                { name: "Open Flap", id: "56-0000000e" },
                { name: "Rub Palms", id: "61-00000017" }
            ]
        },
        {
            name: "OAKINK2",
            tasks: [
                { name: "Erase Board", id: "00-00000153" },
                { name: "Secure Lid", id: "05-0000017c" },
                { name: "Place Object", id: "19-0000001d" },
                { name: "Slide Eraser", id: "60-00000241" }
            ]
        },
        {
            name: "ARCTIC",
            tasks: [
                { name: "Use Scissors", id: "08-0000001b" },
                { name: "Grab Waffleiron", id: "18-0000003a" },
                { name: "Use Laptop", id: "28-0000011c" },
                { name: "Use Phone", id: "63-00000048" }
            ]
        },
        {
            name: "DexYCB",
            tasks: [
                { name: "Grasp Can", id: "22-0000023b" },
                { name: "Move Box", id: "37-0000004c" },
                { name: "Squeeze Clamp", id: "37-0000026b" },
                { name: "Lift Box", id: "40-00000252" },
                { name: "Grasp Box", id: "48-0000020d" },
                { name: "Grasp Bottle", id: "60-0000007e" }
            ]
        },
        {
            name: "Taste-Rob",
            tasks: [
                { name: "Move Eyeliner", id: "03-00000062" },
                { name: "Lift Teapot", id: "22-00000291" },
                { name: "Scoop Spoon", id: "42-000002f2" },
                { name: "Place Food", id: "46-000003cd" }
            ]
        }
    ]
};

// Helper functions to generate paths automatically
function getDatasetThumbnail(datasetName) {
    return `assets/image/dataset/thumb_${datasetName.toLowerCase()}.jpg`;
}

function getTaskVideo(datasetName, taskId) {
    return `assets/video/task/${datasetName}/${taskId}.mp4`;
}

function getTaskThumbnail(datasetName, taskId) {
    return `assets/video/task/${datasetName}/${taskId}.png`;
}

// Demo State Management - Dataset -> Task interaction
let currentNavigation = {
    datasetIndex: null,
    taskIndex: null
};

let datasetSliderOffset = 0;
let taskSliderOffset = 0;
const sliderStep = 250; // pixels per slide (adjusted for larger cards)

// Initialize the demo
function initDemo() {
    renderDatasets();
    renderTasks(); // Initially show placeholder
    resetNavigation();
    
    // Auto-select first dataset
    if (demoData.datasets.length > 0) {
        selectDataset(0);
    }
    
    const totalTasks = demoData.datasets.reduce((total, dataset) => {
        return total + dataset.tasks.length;
    }, 0);
}

// Render datasets in slider
function renderDatasets() {
    const datasetTrack = document.getElementById('datasetTrack');
    if (!datasetTrack) return;
    
    datasetTrack.innerHTML = '';
    
    demoData.datasets.forEach((dataset, index) => {
        const card = document.createElement('div');
        card.className = 'dataset-card';
        card.onclick = () => selectDataset(index);
        
        const thumbnailPath = getDatasetThumbnail(dataset.name);
        card.innerHTML = `
            <div class="dataset-image">
                <img src="${thumbnailPath}" alt="${dataset.name}" onerror="this.src='assets/image/framework.png'">
                <div class="dataset-text-overlay">${dataset.name}</div>
            </div>
            <div class="dataset-name">${dataset.name}</div>
        `;
        
        datasetTrack.appendChild(card);
    });
    
    updateDatasetSliderControls();
}

// Render tasks in slider for selected dataset
function renderTasks(datasetIndex = null) {
    const taskTrack = document.getElementById('taskTrack');
    if (!taskTrack) return;
    
    taskTrack.innerHTML = '';
    
    // If no dataset selected, show placeholder
    if (datasetIndex === null || !demoData.datasets[datasetIndex]) {
        const placeholder = document.createElement('div');
        placeholder.className = 'task-placeholder';
        placeholder.innerHTML = `
            <div class="placeholder-content">
                <div class="placeholder-icon">📋</div>
                <p>Select a dataset to view tasks</p>
            </div>
        `;
        taskTrack.appendChild(placeholder);
        return;
    }
    
    const dataset = demoData.datasets[datasetIndex];
    dataset.tasks.forEach((task, taskIndex) => {
        const card = document.createElement('div');
        card.className = 'task-card';
        card.onclick = () => selectTask(taskIndex);
        
        const thumbnailPath = getTaskThumbnail(dataset.name, task.id);
        card.innerHTML = `
            <div class="task-image">
                <img src="${thumbnailPath}" alt="${task.name}" onerror="this.src='assets/image/framework.png'">
            </div>
            <div class="task-name">${task.name}</div>
        `;
        
        taskTrack.appendChild(card);
    });
    
    updateTaskSliderControls();
}

// Dataset slider controls
function slideDatasets(direction) {
    const viewport = document.querySelector('.dataset-section .slider-viewport');
    const track = document.getElementById('datasetTrack');
    if (!viewport || !track) return;
    
    const viewportWidth = viewport.offsetWidth;
    const trackWidth = track.scrollWidth;
    const maxOffset = Math.max(0, trackWidth - viewportWidth);
    
    // Check if button should be disabled
    const leftBtn = document.getElementById('datasetSliderLeft');
    const rightBtn = document.getElementById('datasetSliderRight');
    
    if (direction === 'left' && leftBtn && !leftBtn.disabled) {
        datasetSliderOffset = Math.max(0, datasetSliderOffset - sliderStep);
    } else if (direction === 'right' && rightBtn && !rightBtn.disabled) {
        datasetSliderOffset = Math.min(maxOffset, datasetSliderOffset + sliderStep);
    }
    
    track.style.transform = `translateX(-${datasetSliderOffset}px)`;
    updateDatasetSliderControls();
}

// Task slider controls
function slideTasks(direction) {
    const viewport = document.querySelector('.task-section .slider-viewport');
    const track = document.getElementById('taskTrack');
    if (!viewport || !track) return;
    
    const viewportWidth = viewport.offsetWidth;
    const trackWidth = track.scrollWidth;
    const maxOffset = Math.max(0, trackWidth - viewportWidth);
    
    // Check if button should be disabled
    const leftBtn = document.getElementById('taskSliderLeft');
    const rightBtn = document.getElementById('taskSliderRight');
    
    if (direction === 'left' && leftBtn && !leftBtn.disabled) {
        taskSliderOffset = Math.max(0, taskSliderOffset - sliderStep);
    } else if (direction === 'right' && rightBtn && !rightBtn.disabled) {
        taskSliderOffset = Math.min(maxOffset, taskSliderOffset + sliderStep);
    }
    
    track.style.transform = `translateX(-${taskSliderOffset}px)`;
    updateTaskSliderControls();
}

function updateDatasetSliderControls() {
    const leftBtn = document.getElementById('datasetSliderLeft');
    const rightBtn = document.getElementById('datasetSliderRight');
    const viewport = document.querySelector('.dataset-section .slider-viewport');
    const track = document.getElementById('datasetTrack');
    
    if (!leftBtn || !rightBtn || !viewport || !track) return;
    
    const viewportWidth = viewport.offsetWidth;
    const trackWidth = track.scrollWidth;
    const maxOffset = Math.max(0, trackWidth - viewportWidth);
    
    // Always show buttons
    leftBtn.style.display = 'flex';
    rightBtn.style.display = 'flex';
    
    // Enable/disable based on current position
    if (trackWidth > viewportWidth) {
        leftBtn.disabled = datasetSliderOffset <= 0;
        rightBtn.disabled = datasetSliderOffset >= maxOffset;
    } else {
        leftBtn.disabled = true;
        rightBtn.disabled = true;
    }
    
    // Update visual state
    leftBtn.style.opacity = leftBtn.disabled ? '0.3' : '1';
    rightBtn.style.opacity = rightBtn.disabled ? '0.3' : '1';
    leftBtn.style.cursor = leftBtn.disabled ? 'default' : 'pointer';
    rightBtn.style.cursor = rightBtn.disabled ? 'default' : 'pointer';
}

function updateTaskSliderControls() {
    const leftBtn = document.getElementById('taskSliderLeft');
    const rightBtn = document.getElementById('taskSliderRight');
    const viewport = document.querySelector('.task-section .slider-viewport');
    const track = document.getElementById('taskTrack');
    
    if (!leftBtn || !rightBtn || !viewport || !track) return;
    
    const viewportWidth = viewport.offsetWidth;
    const trackWidth = track.scrollWidth;
    const maxOffset = Math.max(0, trackWidth - viewportWidth);
    
    // Always show buttons
    leftBtn.style.display = 'flex';
    rightBtn.style.display = 'flex';
    
    // Enable/disable based on current position
    if (trackWidth > viewportWidth) {
        leftBtn.disabled = taskSliderOffset <= 0;
        rightBtn.disabled = taskSliderOffset >= maxOffset;
    } else {
        leftBtn.disabled = true;
        rightBtn.disabled = true;
    }
    
    // Update visual state
    leftBtn.style.opacity = leftBtn.disabled ? '0.3' : '1';
    rightBtn.style.opacity = rightBtn.disabled ? '0.3' : '1';
    leftBtn.style.cursor = leftBtn.disabled ? 'default' : 'pointer';
    rightBtn.style.cursor = rightBtn.disabled ? 'default' : 'pointer';
}

// Select dataset
function selectDataset(datasetIndex) {
    currentNavigation.datasetIndex = datasetIndex;
    currentNavigation.taskIndex = null;
    
    // Update visual selection
    document.querySelectorAll('.dataset-card').forEach((card, index) => {
        if (index === datasetIndex) {
            card.classList.add('selected');
        } else {
            card.classList.remove('selected');
        }
    });
    
    // Render tasks for the selected dataset
    renderTasks(datasetIndex);
    
    // Reset task slider offset when dataset changes
    taskSliderOffset = 0;
    const taskTrack = document.getElementById('taskTrack');
    if (taskTrack) {
        taskTrack.style.transform = 'translateX(0)';
    }
    
    // Clear video display when dataset changes
    const videoDisplay = document.getElementById('videoDisplay');
    if (videoDisplay) {
        videoDisplay.innerHTML = `
            <div class="demo-placeholder">
                <div class="icon">🎬</div>
                <p>Select a task to begin</p>
            </div>
        `;
        videoDisplay.classList.remove('playing');
    }
    
    // Update integrated breadcrumb navigation
    updateIntegratedBreadcrumb();
    
    const dataset = demoData.datasets[datasetIndex];
}

// Select task
function selectTask(taskIndex) {
    // Make sure we have a dataset selected
    if (currentNavigation.datasetIndex === null) return;
    
    currentNavigation.taskIndex = taskIndex;
    
    // Update visual selection
    document.querySelectorAll('.task-card').forEach((card, index) => {
        if (index === taskIndex) {
            card.classList.add('selected');
        } else {
            card.classList.remove('selected');
        }
    });
    
    // Get the task from the current dataset
    const dataset = demoData.datasets[currentNavigation.datasetIndex];
    const task = dataset.tasks[taskIndex];
    
    // Directly play the task video
    if (task) {
        const videoPath = getTaskVideo(dataset.name, task.id);
        showVideo(videoPath, task.name);
    }
    
    // Update integrated breadcrumb navigation
    updateIntegratedBreadcrumb();
    
}

// Update integrated breadcrumb navigation
function updateIntegratedBreadcrumb() {
    const breadcrumbNav = document.getElementById('breadcrumbNav');
    const breadcrumbPath = document.getElementById('breadcrumbPath');
    
    if (!breadcrumbNav || !breadcrumbPath) return;
    
    // Build and display current path
    const pathItems = buildBreadcrumbPath();
    breadcrumbPath.innerHTML = pathItems.length > 0 ? pathItems.join(' › ') : 'Select dataset and task to begin';

    // Show breadcrumb navigation
    breadcrumbNav.style.display = 'block';
}

// Select category at specific level (modular)
function selectCategory(categoryKey, level) {
    // Truncate categories array to current level
    currentNavigation.categories = currentNavigation.categories.slice(0, level);
    // Set category at current level
    currentNavigation.categories[level] = categoryKey;
    
    // Update visual selection for categories
    document.querySelectorAll('.category-btn').forEach(btn => {
        btn.classList.remove('selected');
    });
    
    // Find and select the clicked category button
    const categoryButtons = document.querySelectorAll('.category-btn');
    categoryButtons.forEach(btn => {
        if (btn.textContent.trim() === demoData.scenarios[currentNavigation.scenario].categories[categoryKey].name) {
            btn.classList.add('selected');
        }
    });
    
    // Automatically play the category's video
    const scenario = demoData.scenarios[currentNavigation.scenario];
    const category = scenario.categories[categoryKey];
    if (category && category.video) {
        currentNavigation.selectedVideo = categoryKey;
        showVideo(category.video, category.name);
    }
    
    // Update integrated breadcrumb navigation
    updateIntegratedBreadcrumb();
}

// Backwards compatibility wrapper
function selectCategory1(categoryKey) {
    selectCategory(categoryKey, 0);
}

// Build breadcrumb path display (dataset -> task navigation)
function buildBreadcrumbPath() {
    const pathItems = [];
    
    if (currentNavigation.datasetIndex !== null) {
        const dataset = demoData.datasets[currentNavigation.datasetIndex];
        pathItems.push(`<span class="breadcrumb-item clickable" onclick="navigateFromBreadcrumb('dataset')">${dataset.name}</span>`);
        
        // Add task if selected
        if (currentNavigation.taskIndex !== null) {
            const task = dataset.tasks[currentNavigation.taskIndex];
            if (task) {
                pathItems.push(`<span class="breadcrumb-item">${task.name}</span>`);
            }
        }
    }
    
    return pathItems;
}




// Render categories in scenario-section
function renderCategories(scenarioKey) {
    const categoryContainer = document.getElementById('categoryContainer');
    if (!categoryContainer) return;
    
    const scenario = demoData.scenarios[scenarioKey];
    if (!scenario || !scenario.categories) {
        categoryContainer.innerHTML = `
            <div class="category-selection">
                <span class="category-title">Select Action</span>
                <div class="category-options">
                    <span class="breadcrumb-hint">Select a scene first</span>
                </div>
            </div>
        `;
        categoryContainer.style.display = 'block';
        return;
    }
    
    // Build category buttons (each category has its own video)
    const categoryButtons = Object.keys(scenario.categories).map(categoryKey => {
        const category = scenario.categories[categoryKey];
        return `<button class="category-btn" onclick="selectCategory('${categoryKey}', 0)">${category.name}</button>`;
    }).join('');
    
    categoryContainer.innerHTML = `
        <div class="category-selection">
            <span class="category-title">Select Action</span>
            <div class="category-options">${categoryButtons}</div>
        </div>
    `;
    categoryContainer.style.display = 'block';
}

// Show dataset image function
function showDatasetImage(imageSrc, title) {
    const videoDisplay = document.getElementById('videoDisplay');
    if (!videoDisplay) return;
    
    // Create image element
    const image = document.createElement('img');
    image.src = imageSrc;
    image.alt = title;
    image.style.width = '100%';
    image.style.height = '100%';
    image.style.objectFit = 'cover';
    image.style.borderRadius = 'var(--border-radius-medium)';
    
    // Handle image loading error
    image.onerror = () => {
        image.src = 'assets/image/framework.png';
        // Reset to default aspect ratio on error
        videoDisplay.style.aspectRatio = '16/9';
    };
    
    // Handle image loaded - adjust aspect ratio
    image.onload = () => {
        const aspectRatio = image.naturalWidth / image.naturalHeight;
        videoDisplay.style.aspectRatio = `${aspectRatio}`;
        
        // Ensure smooth transition
        videoDisplay.style.transition = 'all 0.4s var(--ease-out)';
    };
    
    // Clear and add image
    videoDisplay.innerHTML = '';
    videoDisplay.appendChild(image);
    videoDisplay.classList.remove('playing');
    
}

// Show video function
function showVideo(videoSrc, title) {
    const videoDisplay = document.getElementById('videoDisplay');
    if (!videoDisplay) return;
    
    // Create video element
    const video = document.createElement('video');
    video.src = videoSrc;
    video.autoplay = true;
    video.loop = true;
    video.muted = true;
    video.controls = true;
    video.loading = 'lazy';
    video.preload = 'metadata';
    
    // Handle video loading error
    video.onerror = () => {
        videoDisplay.innerHTML = `
            <div class="demo-placeholder">
                <div class="icon">⚠️</div>
                <p>Video not available</p>
                <small>${title}</small>
            </div>
        `;
        // Reset to default aspect ratio on error
        videoDisplay.style.aspectRatio = '16/9';
    };
    
    // Handle video metadata loaded - adjust aspect ratio
    video.onloadedmetadata = () => {
        const aspectRatio = video.videoWidth / video.videoHeight;
        videoDisplay.style.aspectRatio = `${aspectRatio}`;
        
        // Ensure smooth transition
        videoDisplay.style.transition = 'all 0.4s var(--ease-out)';
    };
    
    // Clear and add video
    videoDisplay.innerHTML = '';
    videoDisplay.appendChild(video);
    videoDisplay.classList.add('playing');
    
    // Smooth scroll to center the video after it's displayed
    // Use a shorter delay for immediate response
    setTimeout(() => {
        scrollToVideo();
    }, 100);
    
}

// Smooth scroll to center the demo video in the viewport
function scrollToVideo() {
    // Find the actual video display area
    const videoDisplay = document.getElementById('videoDisplay');
    if (!videoDisplay) return;
    
    // Small delay to ensure video is rendered
    setTimeout(() => {
        // Use the most effective method - scrollIntoView with center positioning
        videoDisplay.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center' 
        });
    }, 50);
}

// Smooth scroll to center the dataset selection area
function scrollToDatasetSelection() {
    // Find the dataset selection area
    const datasetSection = document.querySelector('.dataset-section');
    if (!datasetSection) return;
    
    // Small delay to ensure content is rendered
    setTimeout(() => {
        // Scroll to center the dataset selection area
        datasetSection.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center' 
        });
    }, 50);
}

// Update breadcrumb navigation with clickable functionality
// Legacy function redirects to new integrated breadcrumb system
function updateBreadcrumb() {
    updateIntegratedBreadcrumb();
}

// Navigate back from breadcrumb click (dataset -> task navigation)
function navigateFromBreadcrumb(level) {
    if (level === 'dataset') {
        // Go back to dataset selection (clear task)
        currentNavigation.taskIndex = null;
        
        // Clear task selection
        document.querySelectorAll('.task-card').forEach(card => {
            card.classList.remove('selected');
        });
        
        // Show dataset image if available
        if (currentNavigation.datasetIndex !== null) {
            const dataset = demoData.datasets[currentNavigation.datasetIndex];
            if (dataset) {
                const thumbnailPath = getDatasetThumbnail(dataset.name);
                showDatasetImage(thumbnailPath, dataset.name);
            }
        }
        
        // Scroll to center the dataset selection area
        scrollToDatasetSelection();
    }
    
    updateIntegratedBreadcrumb();
}

// Reset navigation
function resetNavigation() {
    currentNavigation = {
        datasetIndex: null,
        taskIndex: null
    };
    
    // Reset breadcrumb navigation
    const breadcrumbPath = document.getElementById('breadcrumbPath');
    
    if (breadcrumbPath) {
        breadcrumbPath.innerHTML = 'Select dataset and task to begin';
    }
    
    // Reset selections
    document.querySelectorAll('.dataset-card, .task-card').forEach(el => {
        el.classList.remove('selected');
    });
    
    // Reset video display
    const videoDisplay = document.getElementById('videoDisplay');
    if (videoDisplay) {
        videoDisplay.innerHTML = `
            <div class="demo-placeholder">
                <div class="icon">🎬</div>
                <p>Select dataset and task to begin</p>
            </div>
        `;
        videoDisplay.classList.remove('playing');
    }
    
    updateIntegratedBreadcrumb();
}

// Reset demo
function resetDemo() {
    resetNavigation();
    datasetSliderOffset = 0;
    taskSliderOffset = 0;
    
    const datasetTrack = document.getElementById('datasetTrack');
    const taskTrack = document.getElementById('taskTrack');
    
    if (datasetTrack) {
        datasetTrack.style.transform = 'translateX(0)';
    }
    if (taskTrack) {
        taskTrack.style.transform = 'translateX(0)';
    }
    
    updateDatasetSliderControls();
    updateTaskSliderControls();
    
    // Scroll to center the dataset selection area
    scrollToDatasetSelection();
}

// Add trackpad scrolling support to slider (simplified for dataset/task structure)
function initTrackpadScrolling() {
    // Trackpad scrolling for dataset slider
    const datasetViewport = document.querySelector('.dataset-section .slider-viewport');
    if (datasetViewport) {
        datasetViewport.addEventListener('wheel', (e) => {
            if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
                e.preventDefault();
                const direction = e.deltaX > 0 ? 'right' : 'left';
                slideDatasets(direction);
            }
        }, { passive: false });
    }
    
    // Trackpad scrolling for task slider
    const taskViewport = document.querySelector('.task-section .slider-viewport');
    if (taskViewport) {
        taskViewport.addEventListener('wheel', (e) => {
            if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
                e.preventDefault();
                const direction = e.deltaX > 0 ? 'right' : 'left';
                slideTasks(direction);
            }
        }, { passive: false });
    }
}

// Initialize background video with performance optimizations
function initBackgroundVideo() {
    const backgroundVideo = document.querySelector('.background-video');
    if (!backgroundVideo) return;
    
    // Check for reduced motion preference
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) {
        backgroundVideo.style.display = 'none';
        return;
    }
    
    // Performance optimizations
    backgroundVideo.playbackRate = 1.0;
    backgroundVideo.defaultPlaybackRate = 1.0;
    
    // Optimize video for performance
    if ('requestVideoFrameCallback' in backgroundVideo) {
        // Use frame callback for smoother playback if available
        const updateVideo = () => {
            if (!backgroundVideo.paused && !backgroundVideo.ended) {
                backgroundVideo.requestVideoFrameCallback(updateVideo);
            }
        };
        backgroundVideo.requestVideoFrameCallback(updateVideo);
    }
    
    // Intersection Observer to pause video when not visible
    const videoObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                if (backgroundVideo.paused) {
                    backgroundVideo.play().catch(() => {});
                }
            } else {
                if (!backgroundVideo.paused) {
                    backgroundVideo.pause();
                }
            }
        });
    }, { threshold: 0.1 });
    
    videoObserver.observe(backgroundVideo);
    
    // Ensure video plays on iOS and other restrictive browsers
    backgroundVideo.addEventListener('canplaythrough', () => {
        backgroundVideo.play().catch(() => {
            backgroundVideo.style.display = 'none';
        });
    });
    
    // Handle video loading errors
    backgroundVideo.addEventListener('error', () => {
        backgroundVideo.style.display = 'none';
    });
    
    // Throttle resize events that might affect video
    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            // Force video to recalculate dimensions
            backgroundVideo.style.transform = 'translateZ(0)';
        }, 250);
    });
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        initDemo();
        initTrackpadScrolling();
        initVideoLazyLoading(); // Initialize video lazy loading
        initImageLazyLoading(); // Initialize image lazy loading
        initBackgroundVideo(); // Initialize background video
    }, 100);
    
    // Handle window resize for sliders
    window.addEventListener('resize', () => {
        datasetSliderOffset = 0;
        taskSliderOffset = 0;
        
        const datasetTrack = document.getElementById('datasetTrack');
        const taskTrack = document.getElementById('taskTrack');
        
        if (datasetTrack) {
            datasetTrack.style.transform = 'translateX(0)';
        }
        if (taskTrack) {
            taskTrack.style.transform = 'translateX(0)';
        }
        
        updateDatasetSliderControls();
        updateTaskSliderControls();
    });
});

// BibTeX copy functionality
function copyBibTeX(contentId) {
    const bibtexContent = document.getElementById(contentId);
    const text = bibtexContent.textContent;
    
    // Find the corresponding copy button
    const copyBtn = bibtexContent.parentElement.querySelector('.copy-btn');
    
    navigator.clipboard.writeText(text).then(() => {
        const originalText = copyBtn.textContent;
        copyBtn.textContent = 'Copied!';
        copyBtn.style.background = '#1d1d1f';
        copyBtn.style.color = '#ffffff';
        copyBtn.style.borderColor = '#1d1d1f';
        
        setTimeout(() => {
            copyBtn.textContent = originalText;
            copyBtn.style.background = '';
            copyBtn.style.color = '';
            copyBtn.style.borderColor = '';
        }, 2000);
    }).catch(() => {
        // Modern fallback - try using the legacy API if clipboard API fails
        try {
            const textArea = document.createElement('textarea');
            textArea.value = text;
            textArea.style.position = 'fixed';
            textArea.style.opacity = '0';
            document.body.appendChild(textArea);
            textArea.select();
            
            // Use legacy API with warning suppression
            const success = document.execCommand('copy'); // Legacy API for older browsers
            document.body.removeChild(textArea);
            
            if (success) {
                const originalText = copyBtn.textContent;
                copyBtn.textContent = 'Copied!';
                copyBtn.style.background = '#1d1d1f';
                copyBtn.style.color = '#ffffff';
                copyBtn.style.borderColor = '#1d1d1f';
                
                setTimeout(() => {
                    copyBtn.textContent = originalText;
                    copyBtn.style.background = '';
                    copyBtn.style.color = '';
                    copyBtn.style.borderColor = '';
                }, 2000);
            } else {
                throw new Error('Copy command failed');
            }
        } catch (fallbackErr) {
            // Show error to user
            copyBtn.textContent = 'Copy failed';
            copyBtn.style.background = '#86868b';
            copyBtn.style.color = '#ffffff';
            copyBtn.style.borderColor = '#86868b';
            setTimeout(() => {
                copyBtn.textContent = 'Copy';
                copyBtn.style.background = '';
                copyBtn.style.color = '';
                copyBtn.style.borderColor = '';
            }, 2000);
        }
    });
}