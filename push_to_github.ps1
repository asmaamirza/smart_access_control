# =============================================
# Smart Access Control — GitHub Setup Script
# =============================================
# HOW TO USE:
#   1. Open this folder in PowerShell
#   2. Run:  .\push_to_github.ps1
#   3. Enter your GitHub username when asked
# =============================================

$username = Read-Host "Enter your GitHub username"
$repoName = "smart_access_control"
$repoUrl  = "https://github.com/$username/$repoName.git"

Write-Host ""
Write-Host "=== Step 1: Initializing Git ===" -ForegroundColor Cyan
git init

Write-Host ""
Write-Host "=== Step 2: Staging all files ===" -ForegroundColor Cyan
git add .

Write-Host ""
Write-Host "=== Step 3: Creating first commit ===" -ForegroundColor Cyan
git commit -m "Initial commit: Smart Access Control System"

Write-Host ""
Write-Host "=== Step 4: Connecting to GitHub ===" -ForegroundColor Cyan
git remote add origin $repoUrl
git branch -M main

Write-Host ""
Write-Host "=== Step 5: Pushing to GitHub ===" -ForegroundColor Cyan
Write-Host "You may be asked to log in to GitHub in your browser." -ForegroundColor Yellow
git push -u origin main

Write-Host ""
Write-Host "Done! Visit https://github.com/$username/$repoName to see your repo." -ForegroundColor Green
