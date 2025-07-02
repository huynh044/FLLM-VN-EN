class TranslationApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.isTranslating = false;
        this.initializeElements();
        this.bindEvents();
        this.loadModelInfo();
    }

    initializeElements() {
        // Input elements
        this.sourceText = document.getElementById('sourceText');
        this.targetText = document.getElementById('targetText');
        this.translateBtn = document.getElementById('translateBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.swapBtn = document.getElementById('swapBtn');
        
        // Loading and status elements
        this.loadingSpinner = document.getElementById('loadingSpinner');
        this.statusMessage = document.getElementById('statusMessage');
        this.modelInfo = document.getElementById('modelInfo');
        this.characterCount = document.getElementById('characterCount');
        
        // History elements
        this.historyList = document.getElementById('historyList');
        this.clearHistoryBtn = document.getElementById('clearHistoryBtn');
        
        // Initialize translation history
        this.translationHistory = this.loadHistoryFromStorage();
        this.updateHistoryDisplay();
    }

    bindEvents() {
        // Main action buttons
        this.translateBtn.addEventListener('click', () => this.translateText());
        this.clearBtn.addEventListener('click', () => this.clearText());
        this.swapBtn.addEventListener('click', () => this.swapTexts());
        this.clearHistoryBtn.addEventListener('click', () => this.clearHistory());
        
        // Copy button
        const copyBtn = document.getElementById('copy-btn');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => this.copyTranslation());
        }
        
        // Input events
        this.sourceText.addEventListener('input', () => {
            this.updateCharacterCount();
            this.autoResize(this.sourceText);
        });
        
        this.targetText.addEventListener('input', () => {
            this.autoResize(this.targetText);
        });
        
        // Keyboard shortcuts
        this.sourceText.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                this.translateText();
            }
        });
        
        // Auto-translate on paste (with delay)
        this.sourceText.addEventListener('paste', () => {
            setTimeout(() => {
                if (this.sourceText.value.trim() && this.sourceText.value.length < 1000) {
                    this.translateText();
                }
            }, 100);
        });
    }

    async loadModelInfo() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/model-info`);
            if (response.ok) {
                const info = await response.json();
                this.displayModelInfo(info);
            }
        } catch (error) {
            console.warn('Could not load model info:', error);
            this.showStatus('Warning: Could not connect to translation service', 'warning');
        }
    }

    displayModelInfo(info) {
        this.modelInfo.innerHTML = `
            <strong>Model:</strong> ${info.model_name || 'Unknown'} | 
            <strong>Languages:</strong> ${info.source_language || 'VI'} → ${info.target_language || 'EN'} | 
            <strong>Status:</strong> <span class="text-success">Ready</span>
        `;
    }

    async translateText() {
        const text = this.sourceText.value.trim();
        
        if (!text) {
            this.showStatus('Please enter text to translate', 'warning');
            return;
        }

        if (this.isTranslating) {
            return;
        }

        this.isTranslating = true;
        this.setLoadingState(true);
        this.showStatus('Translating...', 'info');

        try {
            // Get translation parameters from UI
            const maxLength = parseInt(document.getElementById('max-length')?.value || '256');
            const numBeams = parseInt(document.getElementById('num-beams')?.value || '4');
            const temperature = parseFloat(document.getElementById('temperature')?.value || '1.0');

            const response = await fetch(`${this.apiBaseUrl}/translate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    max_length: maxLength,
                    num_beams: numBeams,
                    temperature: temperature
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }

            const result = await response.json();
            this.targetText.value = result.translated_text;
            this.autoResize(this.targetText);
            
            // Add to history
            this.addToHistory(text, result.translated_text);
            
            this.showStatus(`Translation completed in ${result.processing_time?.toFixed(2) || 'N/A'}s`, 'success');
            
            // Auto-clear success message after 3 seconds
            setTimeout(() => {
                if (this.statusMessage.classList.contains('alert-success')) {
                    this.clearStatus();
                }
            }, 3000);

        } catch (error) {
            console.error('Translation error:', error);
            this.showStatus(`Translation failed: ${error.message}`, 'danger');
            this.targetText.value = '';
        } finally {
            this.isTranslating = false;
            this.setLoadingState(false);
        }
    }

    async batchTranslate(texts) {
        if (!Array.isArray(texts) || texts.length === 0) {
            throw new Error('Invalid input for batch translation');
        }

        this.setLoadingState(true);
        this.showStatus(`Translating ${texts.length} items...`, 'info');

        try {
            const response = await fetch(`${this.apiBaseUrl}/translate-batch`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    texts: texts,
                    source_language: 'vi',
                    target_language: 'en'
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }

            const result = await response.json();
            this.showStatus('Batch translation completed', 'success');
            return result.translations;

        } catch (error) {
            this.showStatus(`Batch translation failed: ${error.message}`, 'danger');
            throw error;
        } finally {
            this.setLoadingState(false);
        }
    }

    clearText() {
        this.sourceText.value = '';
        this.targetText.value = '';
        this.updateCharacterCount();
        this.autoResize(this.sourceText);
        this.autoResize(this.targetText);
        this.clearStatus();
        this.sourceText.focus();
    }

    copyTranslation() {
        const text = this.targetText.value.trim();
        if (!text) {
            this.showStatus('No translation to copy', 'warning');
            return;
        }

        this.copyToClipboard(text, document.getElementById('copy-btn'))
            .then(() => {
                this.showStatus('Translation copied to clipboard', 'success');
                setTimeout(() => this.clearStatus(), 2000);
            })
            .catch(() => {
                this.showStatus('Failed to copy to clipboard', 'danger');
            });
    }

    swapTexts() {
        const sourceValue = this.sourceText.value;
        const targetValue = this.targetText.value;
        
        this.sourceText.value = targetValue;
        this.targetText.value = sourceValue;
        
        this.updateCharacterCount();
        this.autoResize(this.sourceText);
        this.autoResize(this.targetText);
    }

    setLoadingState(isLoading) {
        this.translateBtn.disabled = isLoading;
        this.loadingSpinner.style.display = isLoading ? 'inline-block' : 'none';
        
        if (isLoading) {
            this.translateBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Translating...';
        } else {
            this.translateBtn.innerHTML = '<i class="fas fa-language me-2"></i>Translate';
        }
    }

    showStatus(message, type = 'info') {
        this.statusMessage.className = `alert alert-${type} mb-3`;
        this.statusMessage.textContent = message;
        this.statusMessage.style.display = 'block';
    }

    clearStatus() {
        this.statusMessage.style.display = 'none';
    }

    updateCharacterCount() {
        const count = this.sourceText.value.length;
        this.characterCount.textContent = `${count} characters`;
        
        // Change color based on length
        if (count > 5000) {
            this.characterCount.className = 'text-danger small';
        } else if (count > 2000) {
            this.characterCount.className = 'text-warning small';
        } else {
            this.characterCount.className = 'text-muted small';
        }
    }

    autoResize(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 300) + 'px';
    }

    addToHistory(sourceText, translatedText) {
        const historyItem = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            source: sourceText,
            target: translatedText
        };

        this.translationHistory.unshift(historyItem);
        
        // Keep only last 50 translations
        if (this.translationHistory.length > 50) {
            this.translationHistory = this.translationHistory.slice(0, 50);
        }

        this.saveHistoryToStorage();
        this.updateHistoryDisplay();
    }

    updateHistoryDisplay() {
        if (this.translationHistory.length === 0) {
            this.historyList.innerHTML = '<p class="text-muted text-center">No translation history yet</p>';
            this.clearHistoryBtn.style.display = 'none';
            return;
        }

        this.clearHistoryBtn.style.display = 'block';
        
        this.historyList.innerHTML = this.translationHistory.map(item => `
            <div class="card mb-2 history-item" data-id="${item.id}">
                <div class="card-body p-3">
                    <div class="d-flex justify-content-between align-items-start mb-2">
                        <small class="text-muted">${this.formatTimestamp(item.timestamp)}</small>
                        <div class="btn-group btn-group-sm">
                            <button class="btn btn-outline-primary btn-sm" onclick="app.useHistoryItem(${item.id})" title="Use this translation">
                                <i class="fas fa-redo"></i>
                            </button>
                            <button class="btn btn-outline-secondary btn-sm" onclick="app.copyToClipboard('${item.target.replace(/'/g, "\\'")}', this)" title="Copy translation">
                                <i class="fas fa-copy"></i>
                            </button>
                            <button class="btn btn-outline-danger btn-sm" onclick="app.removeHistoryItem(${item.id})" title="Delete">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                    <div class="row">
                        <div class="col-md-6">
                            <strong class="text-primary">Vietnamese:</strong>
                            <p class="mb-1 history-text">${this.escapeHtml(item.source)}</p>
                        </div>
                        <div class="col-md-6">
                            <strong class="text-success">English:</strong>
                            <p class="mb-1 history-text">${this.escapeHtml(item.target)}</p>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    useHistoryItem(id) {
        const item = this.translationHistory.find(h => h.id === id);
        if (item) {
            this.sourceText.value = item.source;
            this.targetText.value = item.target;
            this.updateCharacterCount();
            this.autoResize(this.sourceText);
            this.autoResize(this.targetText);
            
            // Scroll to top
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    }

    removeHistoryItem(id) {
        this.translationHistory = this.translationHistory.filter(h => h.id !== id);
        this.saveHistoryToStorage();
        this.updateHistoryDisplay();
    }

    clearHistory() {
        if (confirm('Are you sure you want to clear all translation history?')) {
            this.translationHistory = [];
            this.saveHistoryToStorage();
            this.updateHistoryDisplay();
        }
    }

    async copyToClipboard(text, button) {
        try {
            await navigator.clipboard.writeText(text);
            if (button) {
                const originalIcon = button.innerHTML;
                button.innerHTML = '<i class="fas fa-check me-1"></i>Copied!';
                button.classList.add('btn-success');
                button.classList.remove('btn-outline-success');
                
                setTimeout(() => {
                    button.innerHTML = originalIcon;
                    button.classList.remove('btn-success');
                    button.classList.add('btn-outline-success');
                }, 1500);
            }
            return Promise.resolve();
        } catch (error) {
            console.error('Failed to copy to clipboard:', error);
            return Promise.reject(error);
        }
    }

    loadHistoryFromStorage() {
        try {
            const stored = localStorage.getItem('translationHistory');
            return stored ? JSON.parse(stored) : [];
        } catch (error) {
            console.error('Failed to load history from storage:', error);
            return [];
        }
    }

    saveHistoryToStorage() {
        try {
            localStorage.setItem('translationHistory', JSON.stringify(this.translationHistory));
        } catch (error) {
            console.error('Failed to save history to storage:', error);
        }
    }

    formatTimestamp(isoString) {
        const date = new Date(isoString);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins} min ago`;
        if (diffMins < 1440) return `${Math.floor(diffMins / 60)} hours ago`;
        
        return date.toLocaleDateString();
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Utility method for file upload translation
    async handleFileUpload(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = async (e) => {
                try {
                    const content = e.target.result;
                    const lines = content.split('\n').filter(line => line.trim());
                    
                    if (lines.length > 100) {
                        throw new Error('File too large. Maximum 100 lines supported.');
                    }
                    
                    const translations = await this.batchTranslate(lines);
                    resolve(translations);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new TranslationApp();
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TranslationApp;
}
