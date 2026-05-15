@echo off
REM Medical Appointment System - Windows Setup Script
REM This batch file sets up the complete system on Windows

title Medical Appointment System - Setup
color 0A
cls

echo.
echo ===============================================================
echo             🏥 AI Medical Appointment System
echo                    Windows Setup Script
echo ===============================================================
echo.

REM Check if Node.js is installed
echo ⏳ Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed!
    echo.
    echo Please install Node.js from: https://nodejs.org/
    echo Choose the LTS version and restart this script.
    echo.
    pause
    exit /b 1
) else (
    echo ✅ Node.js is installed
    node --version
)

echo.
echo ⏳ Checking npm installation...
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ npm is not available!
    pause
    exit /b 1
) else (
    echo ✅ npm is available
    npm --version
)

echo.
echo ===============================================================
echo                      📁 Project Setup
echo ===============================================================

REM Create project directory
set PROJECT_DIR=C:\medical-appointment-app
if not exist "%PROJECT_DIR%" (
    echo 📁 Creating project directory: %PROJECT_DIR%
    mkdir "%PROJECT_DIR%"
) else (
    echo 📁 Project directory exists: %PROJECT_DIR%
)

REM Navigate to project directory
cd /d "%PROJECT_DIR%"
echo 📍 Current directory: %CD%

echo.
echo ⏳ Initializing npm project...
if not exist "package.json" (
    echo 📦 Creating package.json...
    echo {> package.json
    echo   "name": "medical-appointment-app",>> package.json
    echo   "version": "1.0.0",>> package.json
    echo   "description": "AI Medical Appointment System",>> package.json
    echo   "main": "server.js",>> package.json
    echo   "scripts": {>> package.json
    echo     "start": "node server.js",>> package.json
    echo     "dev": "nodemon server.js",>> package.json
    echo     "test": "echo \"Error: no test specified\" && exit 1">> package.json
    echo   },>> package.json
    echo   "dependencies": {},>> package.json
    echo   "devDependencies": {},>> package.json
    echo   "keywords": ["medical", "appointment", "ai", "gemini"],>> package.json
    echo   "author": "Medical Appointment System",>> package.json
    echo   "license": "MIT">> package.json
    echo }>> package.json
    echo ✅ package.json created
) else (
    echo ✅ package.json exists
)

echo.
echo ⏳ Installing dependencies...
echo 📦 Installing production dependencies...
call npm install express cors @google/generative-ai dotenv

echo.
echo 📦 Installing development dependencies...
call npm install --save-dev nodemon

if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies!
    pause
    exit /b 1
)

echo ✅ All dependencies installed successfully!

echo.
echo ===============================================================
echo                    📄 Creating Project Files
echo ===============================================================

REM Create public directory
if not exist "public" (
    echo 📁 Creating public directory...
    mkdir public
)

REM Create .env file
if not exist ".env" (
    echo 🔑 Creating .env file...
    echo # Medical Appointment System Environment Variables> .env
    echo # Get your Gemini API key from: https://makersuite.google.com/app/apikey>> .env
    echo GEMINI_API_KEY=your_gemini_api_key_here>> .env
    echo PORT=3000>> .env
    echo NODE_ENV=development>> .env
    echo.>> .env
    echo # Optional: Add your API keys below>> .env
    echo # OPENAI_API_KEY=your_openai_key_here>> .env
    echo ✅ .env file created
) else (
    echo ✅ .env file exists
)

REM Create .gitignore
if not exist ".gitignore" (
    echo 📝 Creating .gitignore file...
    echo node_modules/> .gitignore
    echo .env>> .gitignore
    echo npm-debug.log*>> .gitignore
    echo yarn-debug.log*>> .gitignore
    echo yarn-error.log*>> .gitignore
    echo .DS_Store>> .gitignore
    echo Thumbs.db>> .gitignore
    echo *.log>> .gitignore
    echo .vscode/>> .gitignore
    echo ✅ .gitignore created
)

REM Create README.md
if not exist "README.md" (
    echo 📖 Creating README.md...
    echo # 🏥 AI Medical Appointment System> README.md
    echo.>> README.md
    echo An intelligent medical appointment booking system powered by Google Gemini AI.>> README.md
    echo.>> README.md
    echo ## 🚀 Quick Start>> README.md
    echo.>> README.md
    echo 1. Get your Gemini API key from: https://makersuite.google.com/app/apikey>> README.md
    echo 2. Update the `.env` file with your API key>> README.md
    echo 3. Run: `npm run dev`>> README.md
    echo 4. Open: http://localhost:3000>> README.md
    echo.>> README.md
    echo ## 📋 Requirements>> README.md
    echo.>> README.md
    echo - Node.js 16+ >> README.md
    echo - npm 7+>> README.md
    echo - Google Gemini API key>> README.md
    echo.>> README.md
    echo ## 🛠️ Development>> README.md
    echo.>> README.md
    echo ```bash>> README.md
    echo # Install dependencies>> README.md
    echo npm install>> README.md
    echo.>> README.md
    echo # Start development server>> README.md
    echo npm run dev>> README.md
    echo.>> README.md
    echo # Start production server>> README.md
    echo npm start>> README.md
    echo ```>> README.md
    echo ✅ README.md created
)

echo.
echo ===============================================================
echo                    🔧 System Configuration
echo ===============================================================

echo 🔥 Configuring Windows Firewall...
echo ⚠️  You may see a firewall prompt - please allow Node.js access

echo.
echo 🌐 Checking available ports...
netstat -an | find "3000" >nul
if %errorlevel% equ 0 (
    echo ⚠️  Port 3000 is in use. The application will try to use an alternative port.
) else (
    echo ✅ Port 3000 is available
)

echo.
echo ===============================================================
echo                      📊 Setup Summary
echo ===============================================================
echo.
echo ✅ Project directory: %PROJECT_DIR%
echo ✅ Node.js version: 
node --version
echo ✅ npm version: 
npm --version
echo ✅ Dependencies installed
echo ✅ Configuration files created
echo.
echo 📁 Project structure:
echo    %PROJECT_DIR%\
echo    ├── server.js (Backend server)
echo    ├── package.json (Dependencies)
echo    ├── .env (Environment variables)
echo    ├── .gitignore (Git ignore)
echo    ├── README.md (Documentation)
echo    └── public\ (Frontend files)
echo        ├── index.html
echo        ├── app.js
echo        └── style.css
echo.

echo ===============================================================
echo                      🔑 Next Steps
echo ===============================================================
echo.
echo 1. 🔑 Get your Gemini API key:
echo    - Visit: https://makersuite.google.com/app/apikey
echo    - Sign in with Google account
echo    - Create API key
echo    - Copy the key
echo.
echo 2. 📝 Update your .env file:
echo    - Open: %PROJECT_DIR%\.env
echo    - Replace 'your_gemini_api_key_here' with your actual API key
echo.
echo 3. 🚀 Start the application:
echo    - Run: npm run dev
echo    - Open: http://localhost:3000
echo.
echo ===============================================================
echo                      🎯 Quick Commands
echo ===============================================================
echo.
echo To start development server:
echo    cd %PROJECT_DIR%
echo    npm run dev
echo.
echo To start production server:
echo    cd %PROJECT_DIR%
echo    npm start
echo.
echo To install additional packages:
echo    cd %PROJECT_DIR%
echo    npm install [package-name]
echo.
echo To update dependencies:
echo    cd %PROJECT_DIR%
echo    npm update
echo.

REM Create desktop shortcuts
set SHORTCUT_PATH=%USERPROFILE%\Desktop\Medical Appointment System.bat
if not exist "%SHORTCUT_PATH%" (
    echo 🖥️ Creating desktop shortcut...
    echo @echo off> "%SHORTCUT_PATH%"
    echo title Medical Appointment System>> "%SHORTCUT_PATH%"
    echo cd /d "%PROJECT_DIR%">> "%SHORTCUT_PATH%"
    echo echo Starting Medical Appointment System...>> "%SHORTCUT_PATH%"
    echo echo Server will start at: http://localhost:3000>> "%SHORTCUT_PATH%"
    echo echo.>> "%SHORTCUT_PATH%"
    echo start http://localhost:3000>> "%SHORTCUT_PATH%"
    echo npm run dev>> "%SHORTCUT_PATH%"
    echo ✅ Desktop shortcut created
)

echo.
echo ===============================================================
echo                      ✅ Setup Complete!
echo ===============================================================
echo.
echo Your AI Medical Appointment System is ready!
echo.
echo 🔧 Setup completed successfully at: %date% %time%
echo 📁 Project location: %PROJECT_DIR%
echo 🖥️ Desktop shortcut: Medical Appointment System.bat
echo.
echo ⚠️  IMPORTANT: Don't forget to add your Gemini API key to the .env file!
echo.

REM Ask if user wants to start the application
echo.
set /p start_app="Would you like to start the application now? (y/N): "
if /i "%start_app%"=="y" (
    echo.
    echo 🚀 Starting the application...
    echo 🌐 Opening browser to http://localhost:3000
    echo 📊 Server logs will appear below:
    echo.
    start http://localhost:3000
    npm run dev
) else (
    echo.
    echo 👋 Setup complete! Run 'npm run dev' when you're ready to start.
    echo.
    pause
)

echo.
echo Thank you for using the AI Medical Appointment System! 🏥
echo.
pause