# File: README_GDRIVE_SETUP.md
# Google Drive Setup Guide for Stock Scanner

This guide will help you set up Google Drive integration for the Stock Scanner application.

## Prerequisites

- A Google account
- Access to Google Cloud Console
- Python 3.8 or higher installed

## Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Enter project name: "Stock Scanner"
4. Click "Create"

## Step 2: Enable Google Drive API

1. In your project, go to "APIs & Services" → "Library"
2. Search for "Google Drive API"
3. Click on it and press "Enable"

## Step 3: Create OAuth 2.0 Credentials

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "OAuth client ID"
3. If prompted, configure OAuth consent screen:
   - User Type: External
   - App name: "Stock Scanner"
   - User support email: Your email
   - Developer contact: Your email
   - Click "Save and Continue"
   - Scopes: Skip this (click "Save and Continue")
   - Test users: Add your email
   - Click "Save and Continue"
4. Back to Create OAuth client ID:
   - Application type: "Desktop app"
   - Name: "Stock Scanner Desktop"
   - Click "Create"
5. Download the JSON file
6. Rename it to `credentials.json`
7. Place it in your application root directory

## Step 4: Get Google Drive Folder IDs

### For Historical_Data folder:
1. Open your Historical_Data folder in Google Drive
2. Look at the URL: `https://drive.google.com/drive/folders/[FOLDER_ID]`
3. Copy the FOLDER_ID part

### For EOD_Data folder:
1. Open your EOD_Data folder in Google Drive
2. Copy the FOLDER_ID from the URL

## Step 5: Configure .env File

1. Copy `.env.example` to `.env`:
```bash
   cp .env.example .env