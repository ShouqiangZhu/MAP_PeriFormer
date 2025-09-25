# MAP-PeriFormer: Transformer-based Intraoperative Hypotension Prediction

This repository contains the official implementation of **MAP-PeriFormer**, a Transformer-based model for real-time prediction of intraoperative mean arterial pressure (MAP) and hypotension events, as described in our paper *"MAP-PeriFormer: A Transformer-Based Big Data Model for Real-Time Intraoperative Mean Arterial Pressure Prediction with External Validation"*.

## 📖 Overview

MAP-PeriFormer is a deep learning model that predicts:
- Continuous MAP values at 3, 5, and 10-minute horizons
- Binary hypotension events (MAP < 65 mmHg)
- With state-of-the-art performance on large-scale clinical datasets

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ShouqiangZhu/IOH_Transformer.git
cd IOH_Transformer
pip install -e .
