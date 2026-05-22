from setuptools import setup
import os

APP = ['murmur.py']
DATA_FILES = []
BUNDLE_ID = os.environ.get('BUNDLE_ID', 'com.canopystudio.murmur')
OPTIONS = {
    'argv_emulation': False,
    'plist': {
        'CFBundleName': 'Murmur',
        'CFBundleDisplayName': 'Murmur',
        'CFBundleIdentifier': BUNDLE_ID,
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'LSUIElement': True,  # Hide from dock (menu bar app)
        'NSMicrophoneUsageDescription': 'Murmur needs microphone access to record your voice for transcription.',
        'NSAppleEventsUsageDescription': 'Murmur needs accessibility access to type transcribed text.',
    },
    'packages': [
        'rumps',
        'sounddevice', 
        'numpy',
        'whisper',
        'pyperclip',
        'scipy',
        'pynput',
        'torch',
        'tiktoken',
        'numba',
    ],
    'includes': [
        'cffi',
        'AppKit',
        'Foundation',
    ],
    'excludes': ['tkinter'],
    'iconfile': 'Murmur.icns',
}

setup(
    app=APP,
    name='Murmur',
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
