# Secure IoT Sensor Anomaly Detection

Defensive anomaly-detection project for IoT sensor telemetry with simulated spoofing and false-data injection scenarios.

This project connects IoT sensing, cyber-physical systems, machine learning, and security. The goal is to study how suspicious sensor behavior can be detected before it affects monitoring, automation, or digital-twin decisions.

## Project Motivation

IoT systems depend on sensor streams to understand the physical world. If a sensor value is spoofed, replayed, frozen, or manipulated, the software layer may make incorrect decisions.

This repository studies a defensive question:

    Can abnormal IoT sensor behavior be detected from telemetry data?

## Dataset

The project uses a real IoT telemetry dataset downloaded from Kaggle:

    Environmental Sensor Telemetry Data

The raw dataset contains sensor streams with fields such as:

- timestamp
- device identifier
- carbon monoxide
- humidity
- light state
- LPG
- motion state
- smoke
- temperature

The raw dataset is not committed to this repository. Place it locally at:

    data/raw/iot_telemetry_data.csv

## Planned Workflow

1. Inspect real IoT telemetry data
2. Clean and prepare the sensor stream
3. Inject controlled spoofing/anomaly scenarios
4. Build rule-based anomaly detection
5. Train simple machine-learning detectors
6. Evaluate with confusion matrix, precision, recall, and F1-score
7. Visualize anomaly timelines and model results

## Defensive Scope

This project is focused on defensive monitoring and anomaly detection. It does not provide instructions for attacking real networks, devices, or unauthorized systems.

## Repository Structure

    secure-iot-sensor-anomaly-detection/
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── docs/
    ├── results/
    ├── src/
    │   └── inspect_dataset.py
    ├── .gitignore
    ├── requirements.txt
    └── README.md

## Current Status

Initial project structure and real dataset inspection are implemented. Spoofing injection and detection models will be added in later steps.
