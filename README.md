---
title: Token length analysis for FLORES200.
emoji: 🐨
colorFrom: gray
colorTo: indigo
sdk: streamlit
sdk_version: 1.26.0
app_file: app.py
pinned: false
license: mit
---

# Tokenization length analysis for FLORES200

## About
This dashboard is based on the app of @yenniejun https://github.com/yenniejun/tokenizers-languages.git. The FLORES200.tok.csv is obtained through this repo: https://github.com/isaacnicolas/open-nllb-misc.git  
This repository hosts a Streamlit-based [web application](https://huggingface.co/spaces/isaacnicolas/TokAnalysisFLORES200) designed to showcase the tokenization process for the FLORES-200 languages. FLORES-200 is a comprehensive benchmark for multilingual translation covering 200+ languages. This app provides an interactive platform for users to visualize and understand how tokenization works across different languages in this extensive dataset.

## Features
- Basic statistics (mean, std, median, min, max)
- Distribution plots
- Sample texts
- Most common tokens (max. 50 per language)

## TO DO
Maybe add pages with extensive analysis for different language families such as HBS.