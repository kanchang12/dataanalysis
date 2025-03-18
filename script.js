// Global variable to store filename
let currentFilename = null;

// Document ready function
document.addEventListener('DOMContentLoaded', function () {
    // File input handling
    const fileInput = document.getElementById('file-upload');
    const fileLabel = document.querySelector('.file-label');
    const fileName = document.getElementById('file-name');

    fileInput.addEventListener('change', function () {
        if (this.files && this.files[0]) {
            fileName.textContent = this.files[0].name;
        } else {
            fileName.textContent = 'No file chosen';
        }
    });

    // Initialize chat messages container
    const chatContent = document.getElementById('chat-content');

    // Add a welcome message
    addChatMessage('AI Assistant', 'Welcome! Upload a CSV file, and I can help you analyze it. You can also ask me questions about the data.');

    // Make sure details elements are open by default
    document.querySelectorAll('details').forEach(detail => {
        detail.setAttribute('open', '');
    });
});

// Function to add chat messages
function addChatMessage(sender, message) {
    const chatContent = document.getElementById('chat-content');
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message');

    if (sender === 'User') {
        messageElement.classList.add('user-message');
    } else {
        messageElement.classList.add('ai-message');
    }

    messageElement.innerHTML = `
        <div class="message-header">${sender}</div>
        <div class="message-body">${message}</div>
    `;

    chatContent.appendChild(messageElement);
    chatContent.scrollTop = chatContent.scrollHeight;
}

// Function to update statistics section
function updateStatistics(stats) {
    const statsContent = document.getElementById('stats-content');
    
    // Clear previous stats
    statsContent.innerHTML = '';
    
    if (!stats || Object.keys(stats).length === 0) {
        statsContent.innerHTML = '<p>No statistics available yet. Upload and analyze data first.</p>';
        return;
    }
    
    // Create stats header
    const header = document.createElement('h3');
    header.textContent = 'Summary Statistics';
    statsContent.appendChild(header);
    
    // Create key metrics row for numerical summaries if available
    if (stats.summary && Object.keys(stats.summary).length > 0) {
        const metricsRow = document.createElement('div');
        metricsRow.className = 'metrics-row';
        
        // Convert summary object to array of {label, value} objects
        const metrics = Object.entries(stats.summary).map(([key, value]) => {
            return {
                label: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                value: typeof value === 'number' ? value.toFixed(2) : value
            };
        });
        
        metrics.forEach(metric => {
            const div = document.createElement('div');
            div.className = 'key-metric';
            
            const label = document.createElement('span');
            label.className = 'label';
            label.textContent = metric.label;
            
            const value = document.createElement('span');
            value.className = 'value';
            value.textContent = metric.value;
            
            div.appendChild(label);
            div.appendChild(value);
            metricsRow.appendChild(div);
        });
        
        statsContent.appendChild(metricsRow);
    }
    
    // Create detailed stats table if available
    if (stats.details && stats.details.length > 0) {
        const table = document.createElement('table');
        
        // Get headers from the first row
        const headers = Object.keys(stats.details[0]);
        
        const tableHead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        
        headers.forEach(header => {
            const th = document.createElement('th');
            th.textContent = header.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            headerRow.appendChild(th);
        });
        
        tableHead.appendChild(headerRow);
        table.appendChild(tableHead);
        
        const tableBody = document.createElement('tbody');
        
        stats.details.forEach(row => {
            const tr = document.createElement('tr');
            
            headers.forEach(header => {
                const td = document.createElement('td');
                let value = row[header];
                
                // Format numbers and percentages
                if (typeof value === 'number') {
                    if (header.includes('percent') || header.includes('rate')) {
                        value = value.toFixed(2) + '%';
                    } else {
                        value = value.toLocaleString();
                    }
                }
                
                // Add color to growth values
                if (header.includes('growth') || header.includes('change')) {
                    if (value > 0) {
                        td.style.color = 'green';
                        value = '+' + value;
                    } else if (value < 0) {
                        td.style.color = 'red';
                    }
                }
                
                td.textContent = value;
                tr.appendChild(td);
            });
            
            tableBody.appendChild(tr);
        });
        
        table.appendChild(tableBody);
        statsContent.appendChild(table);
    }
}

// Handle the analyze button click
document.getElementById('analyze-button').addEventListener('click', function () {
    const fileInput = document.getElementById('file-upload');
    const file = fileInput.files[0];

    if (!file) {
        showNotification('Please select a file first.', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Show loading state
    document.getElementById('analyze-button').disabled = true;
    document.getElementById('analyze-button').innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

    // Upload file to the server
    fetch('/', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                console.log('File uploaded:', data.message);
                currentFilename = file.name; // Store filename

                // Call backend for analysis
                return fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ filename: file.name })
                });
            } else {
                throw new Error(data.error || 'Unknown error during upload');
            }
        })
        .then(response => response.json())
        .then(analysis => {
            if (analysis.error) {
                showNotification(analysis.error, 'error');
            } else {
                // Update the visualization panel with generated graphs
                if (analysis.graphs && analysis.graphs.length > 0) {
                    const graphsContainer = document.getElementById('plotly-graphs-container');
                    if (graphsContainer) { // Check if the container exists
                        graphsContainer.innerHTML = ''; // Clear previous graphs

                        analysis.graphs.forEach(graphData => {
                            const graphDiv = document.createElement('div');
                            graphsContainer.appendChild(graphDiv);
                            Plotly.newPlot(graphDiv, JSON.parse(graphData.graph).data, JSON.parse(graphData.graph).layout);
                        });
                    } else {
                        console.error("plotly-graphs-container not found!");
                    }
                } else {
                    document.getElementById('plotly-graphs-container').innerHTML = '<p>No graphs to display.</p>';
                }

                // Update statistics panel if statistics are available
                if (analysis.statistics) {
                    updateStatistics(analysis.statistics);
                } else {
                    // Create sample statistics if none provided
                    updateStatistics({
                        summary: {
                            count: analysis.data_summary?.row_count || 0,
                            mean: analysis.data_summary?.mean || 0,
                            median: analysis.data_summary?.median || 0,
                            std_dev: analysis.data_summary?.std_dev || 0,
                            min: analysis.data_summary?.min || 0,
                            max: analysis.data_summary?.max || 0,
                        },
                        details: analysis.data_details || []
                    });
                }

                // Display success message
                showNotification('Data analysis complete!', 'success');

                // Add an AI message about the analysis with key points only
                addChatMessage(' ', ` :
                <ul>
                    ${analysis.key_points ? analysis.key_points.map(point => `<li>${point}</li>`).join('') : 
                    '<li>Data has been processed successfully</li><li>Check the visualization for trends</li><li>Detailed statistics are available below</li>'}
                </ul>
                You can explore the visualization and ask me specific questions about your dataset.`);
            }
        })
        .catch(error => {
            console.error('Error during analysis:', error);
            showNotification(error.message, 'error');
        })
        .finally(() => {
            // Reset button state
            document.getElementById('analyze-button').disabled = false;
            document.getElementById('analyze-button').innerHTML = '<i class="fas fa-search"></i> Analyze';
        });
});

// Handle the send button click for chat
document.getElementById('send-button').addEventListener('click', sendChatMessage);

// Handle pressing Enter in the chat input
document.getElementById('chat-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendChatMessage();
    }
});

// Function to send chat messages
function sendChatMessage() {
    const chatInput = document.getElementById('chat-input');
    const userInput = chatInput.value.trim();

    if (!userInput) return;

    // Add user message to chat
    addChatMessage('User', userInput);

    // Clear input
    chatInput.value = '';

    // Show thinking indicator
    const thinkingElement = document.createElement('div');
    thinkingElement.classList.add('chat-message', 'ai-message', 'thinking');
    thinkingElement.innerHTML = `
        <div class="message-header">AI Assistant</div>
        <div class="message-body"><i class="fas fa-spinner fa-spin"></i> Thinking...</div>
    `;
    document.getElementById('chat-content').appendChild(thinkingElement);
    document.getElementById('chat-content').scrollTop = document.getElementById('chat-content').scrollHeight;

    // If no file is uploaded yet, respond with a prompt
    if (!currentFilename) {
        setTimeout(() => {
            document.getElementById('chat-content').removeChild(thinkingElement);
            addChatMessage('AI Assistant', 'Please upload a data file first so I can help you analyze it.');
        }, 1000);
        return;
    }

// In the sendChatMessage function, modify the fetch response handling
fetch('/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question: userInput, filename: currentFilename })
})
    .then(response => response.json())
    .then(data => {
        // Remove thinking indicator
        document.getElementById('chat-content').removeChild(thinkingElement);

        if (data.error) {
            addChatMessage('AI Assistant', `Sorry, I encountered an error: ${data.error}`);
        } else {
            // Display the AI's response
            addChatMessage('AI Assistant', data.analysis);
        }
    })
    .catch(error => {
        // Remove thinking indicator
        document.getElementById('chat-content').removeChild(thinkingElement);
        console.error('Error during chat:', error);
        addChatMessage('AI Assistant', 'Sorry, I encountered an error processing your request.');
    });
}

// Function to show notifications
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.classList.add('notification', `notification-${type}`);
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
            <span>${message}</span>
        </div>
        <button class="notification-close"><i class="fas fa-times"></i></button>
    `;

    // Add to body
    document.body.appendChild(notification);

    // Show with animation
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);

    // Auto-hide after 5 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 5000);

    // Close button
    notification.querySelector('.notification-close').addEventListener('click', () => {
        notification.classList.remove('show');
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    });
}