# 🎓 Student Performance Analysis & Feedback System

A comprehensive web application built with Streamlit that analyzes student performance data from competitive exams (JEE/NEET) and generates detailed reports with AI-powered feedback.

## 🌟 Features

### 📊 Comprehensive Performance Analysis
- **Overall Performance Metrics**: Score, accuracy, time efficiency analysis
- **Subject-wise Breakdown**: Detailed performance across Physics, Chemistry, Mathematics
- **Chapter-level Insights**: Identify strong and weak areas within each subject
- **Difficulty Analysis**: Performance breakdown by question difficulty levels
- **Time Management Analysis**: Identify time traps and optimization opportunities

### 🤖 AI-Powered Feedback
- **Intelligent Coaching**: Professional analysis with actionable recommendations
- **Multiple LLM Support**: Choose between Gemini and Groq for AI feedback
- **Personalized Insights**: Data-driven feedback tailored to individual performance
- **Strategic Recommendations**: Concrete steps for improvement

### 📈 Visual Analytics
- **Interactive Charts**: Overall score distribution, subject-wise performance
- **Time vs Performance Analysis**: Identify question-level patterns
- **Accuracy Visualization**: Subject-wise accuracy comparisons

### 📄 Professional PDF Reports
- **Comprehensive Reports**: Complete analysis with charts and recommendations
- **Professional Layout**: Clean, structured format suitable for students and educators
- **Downloadable**: Easy export for offline review and sharing

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mathongo
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys (Optional - for AI feedback)**
   
   Create a `.env` file or set environment variables:
   ```bash
   # For Gemini AI
   export GEMINI_API_KEY="your-gemini-api-key"
   
   # For Groq AI (Recommended)
   export GROQ_API_KEY="your-groq-api-key"
   ```

4. **Add sample data (Optional)**
   ```bash
   # Place your sample JSON file as:
   # sample_submission_analysis_2.json
   ```

5. **Run the application**
   ```bash
   streamlit run promptly.py
   ```

6. **Open in browser**
   - The app will automatically open at `http://localhost:8501`

## 🎯 Usage Guide

### Data Upload
1. **JSON Format**: Upload performance data in the supported JSON format
2. **Sample Data**: Use the built-in sample data to explore features
3. **File Structure**: Ensure your JSON follows the expected schema

### Analysis Process
1. **Configure Settings**: Choose AI provider and enter API keys
2. **Upload Data**: Select your performance JSON file
3. **Run Analysis**: Click "Analyze Performance" to process data
4. **Review Results**: Explore the generated insights and visualizations

### AI Feedback Configuration
- **Gemini**: Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Groq**: Get API key from [Groq Console](https://console.groq.com)
- **Offline Mode**: Use template feedback when AI is disabled

### PDF Report Generation
- **Automatic Generation**: PDF created on-demand when analysis is complete
- **Download**: Use the download button to save the complete report
- **Professional Format**: Includes all charts, analysis, and recommendations

## 📋 JSON Data Format

Your performance data should follow this structure:

```json
[
  {
    "userId": {"name": "Student Name"},
    "test": {
      "title": "Test Name",
      "totalQuestions": 90,
      "totalMarks": 360,
      "totalTime": 180,
      "syllabus": "<h1>Test Title</h1>..."
    },
    "totalAttempted": 75,
    "totalCorrect": 60,
    "totalMarkScored": 240,
    "totalTimeTaken": 9600,
    "subjects": [...],
    "sections": [...]
  }
]
```

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Data Processing**: Python, NumPy, Pandas-like operations
- **Visualization**: Matplotlib
- **PDF Generation**: ReportLab
- **AI Integration**: Google Generative AI, Groq
- **HTML Parsing**: BeautifulSoup4

## 📦 Dependencies

```txt
streamlit>=1.28.0
matplotlib>=3.5.0
numpy>=1.21.0
beautifulsoup4>=4.11.0
reportlab>=3.6.0
google-generativeai>=0.3.0
groq>=0.4.0
```

## 🎨 Features Overview

### Performance Metrics
- **Score Analysis**: Total marks, percentage, efficiency calculations
- **Accuracy Tracking**: Overall and subject-wise accuracy metrics
- **Time Management**: Time per question, efficiency analysis
- **Attempt Patterns**: Question selection and completion analysis

### AI Coaching Features
- **Professional Analysis**: Expert-level performance evaluation
- **Strategic Recommendations**: Actionable improvement plans
- **Conceptual Gap Analysis**: Identify knowledge gaps
- **Study Strategy**: Personalized preparation guidance

### Visualization Suite
- **Donut Charts**: Score distribution visualization
- **Bar Charts**: Subject-wise performance comparison
- **Time Analysis**: Question-level time management insights
- **Accuracy Plots**: Performance accuracy across subjects

## 🔧 Configuration Options

### LLM Provider Settings
- **Provider Selection**: Choose between Gemini and Groq
- **Model Configuration**: Specify custom models if needed
- **Fallback Mode**: Template feedback when AI is unavailable

### Chart Generation
- **Style Customization**: Professional chart styling
- **Export Quality**: High-resolution chart generation
- **Format Options**: PNG output for PDF integration

### PDF Customization
- **Layout Settings**: Professional report formatting
- **Content Sections**: Configurable report components
- **Branding**: Customizable headers and styling

## 🚨 Troubleshooting

### Common Issues

1. **PDF Generation Fails**
   - Check ReportLab installation
   - Verify chart file paths exist
   - Ensure sufficient disk space

2. **AI Feedback Not Working**
   - Verify API keys are correctly set
   - Check internet connectivity
   - Confirm API quotas/limits

3. **Chart Generation Errors**
   - Ensure matplotlib backend is properly configured
   - Check data structure completeness
   - Verify output directory permissions

4. **JSON Upload Issues**
   - Validate JSON format
   - Check file encoding (UTF-8)
   - Ensure all required fields are present

### Performance Tips
- Use Groq for faster AI responses
- Enable chart caching for repeated analysis
- Process smaller datasets for faster rendering

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues, questions, or feature requests:
- Create an issue in the repository
- Check the troubleshooting section
- Review the usage guide

## 🎓 Educational Context

This application is specifically designed for:
- **JEE Main & Advanced**: Complete analysis for engineering entrance preparation
- **NEET**: Medical entrance exam performance evaluation
- **Mock Tests**: Practice test analysis and improvement tracking
- **Coaching Institutes**: Student performance monitoring and feedback

---

