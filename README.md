# Medical AI Classification & Report Generation System

## System Overview

This application combines deep learning image classification with AI-powered report generation to assist in medical diagnostics. The system classifies medical images to detect and locate diseases, then uses a Small Language Model (SLM) to generate comprehensive medical reports based on the findings.

## Architecture

### Backend (`main.py`)
- **API Framework**: FastAPI
- **Classification Model**: Pre-trained model for disease detection and localization
- **Report Generation**: Qwen2.5-15B via Hugging Face API
- **Deployment**: Dockerized and deployed on AWS App Runner


<img width="875" height="498" alt="aws" src="https://github.com/user-attachments/assets/e8b1cd46-01e2-4b49-9d97-690e8fafdfc8" />


<img width="1783" height="722" alt="cloudwatch aws" src="https://github.com/user-attachments/assets/3e341f1f-18e8-4f94-aa5b-806aa1df9543" />


### Frontend
- **Deployment**: Vercel
- **Communication**: REST API calls to backend endpoints


## Local Setup

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create a `.env` file in the backend directory:
   ```env
   QWEN_MODEL=<your-hugging-face-api-key>
   ```

   To obtain a Hugging Face API key:
   - Visit [Hugging Face](https://huggingface.co/)
   - Sign up or log in to your account
   - Navigate to Settings → Access Tokens
   - Create a new token with read permissions
   - Copy the token to your `.env` file

4. **Run the backend server**
   ```bash
   uvicorn main:app --reload
   ```

   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to the frontend directory**
   ```bash
   cd frontend
   ```

2. **Update API endpoint**
   
   Open the HTML file and locate the JavaScript section where the API URL is defined. Change the endpoint from the deployed AWS App Runner URL to your local backend:
   ```javascript
   // Change from:
   const API_URL = "https://your-app.awsapprunner.com";
   
   // To:
   const API_URL = "http://localhost:8000";
   ```

3. **Open in browser**
   
   Simply open the `index.html` file directly in your web browser (double-click the file or right-click → Open with → Browser)

## Deployment

### Backend Deployment (AWS App Runner)

1. **Build and test Docker image locally**
   ```bash
   docker build -t medical-ai-backend .
   docker run -p 8000:8000 medical-ai-backend
   ```

2. **Deploy to AWS App Runner**
   - Push your Docker image to Amazon ECR or connect your GitHub repository
   - Create a new App Runner service
   - Configure environment variables (add your `QWEN_MODEL` API key)
   - Deploy and obtain your service URL

### Frontend Deployment (Vercel)

1. **Update API endpoint**
   
   Open your HTML file and change the API URL to your AWS App Runner deployment URL:
   ```javascript
   const API_URL = "https://your-app.awsapprunner.com";
   ```

2. **Deploy to Vercel**
   - Push your frontend code to a GitHub repository
   - Connect your repository to Vercel
   - Vercel will automatically deploy your static HTML site
   
   Or use Vercel CLI:
   ```bash
   vercel
   ```

## API Endpoints

### POST `/analyze`
Accepts medical images and returns:
- Disease classification results
- Location/region of detected abnormalities
- Confidence scores
- AI-generated medical report

## Workflow

1. User uploads medical image through frontend
2. Frontend sends image to backend `/analyze` endpoint
3. Backend loads trained classification model
4. Model processes image and identifies disease + location
5. Classification results are passed to Qwen2.5-15B SLM
6. SLM generates comprehensive medical report
7. Report is returned to frontend and displayed to user
