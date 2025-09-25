#!/usr/bin/env python3
"""
Training script for MAP-PeriFormer model
"""

import torch
import yaml
import argparse
from pathlib import Path

from src.ioh_transformer.data.loader import AnesthesiaDataLoader
from src.ioh_transformer.models.transformer import MAPPeriFormer
from src.ioh_transformer.training.trainer import ModelTrainer
from src.ioh_transformer.utils.config_loader import load_config

def main():
    parser = argparse.ArgumentParser(description='Train MAP-PeriFormer model')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Directory to save model outputs')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize data loader
    print("Loading data...")
    data_loader = AnesthesiaDataLoader(args.data_path, config)
    train_loader, val_loader = data_loader.get_data_loaders()
    
    # Initialize model
    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MAPPeriFormer(config['model']).to(device)
    
    # Initialize trainer
    trainer = ModelTrainer(model, config['training'], device)
    
    # Start training
    print("Starting training...")
    trainer.train(train_loader, val_loader, args.output_dir)
    
    print("Training completed!")

if __name__ == '__main__':
    main()
