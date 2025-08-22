import React, { useState, useRef, useEffect } from 'react';
import './App.css';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
}

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  uploadedAt: Date;
}

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files) return;

    Array.from(files).forEach(file => {
      const newFile: UploadedFile = {
        id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
        name: file.name,
        size: file.size,
        type: file.type,
        uploadedAt: new Date()
      };

      setUploadedFiles(prev => [...prev, newFile]);

      // Add a system message about the upload
      const systemMessage: Message = {
        id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
        text: `File "${file.name}" has been uploaded successfully. You can now ask questions about this document.`,
        sender: 'assistant',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, systemMessage]);
    });

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const simulateLLMResponse = async (userMessage: string): Promise<string> => {
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));

    // Simple response simulation based on user input
    const responses = [
      `I understand you're asking about: "${userMessage}". Based on the uploaded files, here's what I can tell you...`,
      `That's an interesting question about "${userMessage}". Let me analyze the content from your documents...`,
      `Regarding "${userMessage}", I can provide insights from the uploaded files...`,
      `I've processed your question about "${userMessage}" and found relevant information in your documents...`
    ];

    return responses[Math.floor(Math.random() * responses.length)];
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
      text: inputMessage,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await simulateLLMResponse(inputMessage);

      const assistantMessage: Message = {
        id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
        text: response,
        sender: 'assistant',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
        text: 'Sorry, I encountered an error processing your request. Please try again.',
        sender: 'assistant',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    }

    setIsLoading(false);
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const removeFile = (fileId: string) => {
    setUploadedFiles(prev => prev.filter(file => file.id !== fileId));
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Document Chat Assistant</h1>
        <p>Upload your files and chat with AI about their content</p>
      </header>

      <div className="app-content">
        <div className="sidebar">
          <div className="file-upload-section">
            <h3>Upload Files</h3>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              multiple
              accept=".pdf,.txt,.doc,.docx,.md"
              style={{ display: 'none' }}
            />
            <button
              className="upload-button"
              onClick={() => fileInputRef.current?.click()}
            >
              📁 Choose Files
            </button>
          </div>

          <div className="uploaded-files">
            <h4>Uploaded Files ({uploadedFiles.length})</h4>
            {uploadedFiles.length === 0 ? (
              <p className="no-files">No files uploaded yet</p>
            ) : (
              <div className="file-list">
                {uploadedFiles.map(file => (
                  <div key={file.id} className="file-item">
                    <div className="file-info">
                      <span className="file-name">{file.name}</span>
                      <span className="file-size">{formatFileSize(file.size)}</span>
                    </div>
                    <button
                      className="remove-file"
                      onClick={() => removeFile(file.id)}
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="chat-container">
          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <h3>Welcome! 👋</h3>
                <p>Upload some files and start asking questions about their content.</p>
              </div>
            ) : (
              messages.map(message => (
                <div
                  key={message.id}
                  className={`message ${message.sender === 'user' ? 'user-message' : 'assistant-message'}`}
                >
                  <div className="message-content">
                    <p>{message.text}</p>
                    <span className="message-time">
                      {message.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="message assistant-message">
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input-container">
            <div className="chat-input">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask a question about your uploaded files..."
                rows={1}
                disabled={isLoading}
              />
              <button
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || isLoading}
                className="send-button"
              >
                📤
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;