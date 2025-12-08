# IRL Streaming Ecosystem Dashboard
**Authors:** Alex Chen Hsieh & Derek Li

**Course:** CS 415: Social Media Data Science Pipelines (Binghamton University)

## Overview
This project investigates the "In Real Life" (IRL) streaming ecosystem, focusing on creators who maintain a dual presence on Twitch and YouTube.

Using a robust data pipeline, we collected weeks of data including stream metadata, video metrics, and user comments. This interactive dashboard allows users to explore the statistical relationships between these platforms, analyze community toxicity, and identify engagement drivers.

🔗 [Streamlit Live Demo](https://alexmicael-irl-ecosystem-dashboard-app-wgmceg.streamlit.app/)

<div align="center">
  <img src="assets\dashboard_home.png" width="100%" alt="Streamlit Dashboard Screenshot" />
</div>


## Research Questions
This dashboard allows you to interactively explore the answers to our three primary research questions:

- **Temporal Toxicity (RQ1):** How does toxicity evolve over time in response to creator-specific events?

- **Cross-Platform Predictability (RQ2):** To what extent do Twitch engagement metrics predict YouTube engagement outcomes?

- **Content Themes (RQ3):** How do specific content keywords affect engagement and reveal related themes?

## How It Works
### The Data Pipeline
1. **Collection:** Data was collected continuously using the Twitch Helix API (every 15 mins) and YouTube Data API v3 (every 4 hours).

2. **Processing:** Comments were analyzed using the Google Perspective API to assign toxicity scores.

3. **Storage:** The dataset is stored in optimized Parquet format for high-performance loading in this dashboard.

## How to Run (Locally)
If you want to run this dashboard on your own machine:

1. **Clone the repository:**

    ``` sh
    git clone https://github.com/AlexMicael/IRL-Ecosystem-Dashboard.git
    cd IRL-Ecosystem-Dashboard
    ```

2. **Install dependencies:**

    ``` sh
    pip install -r requirements.txt
    ```

3. **Run the app:**

    ``` sh
    streamlit run app.py
    ```

## Acknowledgments

This work was supported by Professor Yang at Binghamton University. We also thank the Computer Science Department for providing the virtual machine resources used for the data collection pipeline.
