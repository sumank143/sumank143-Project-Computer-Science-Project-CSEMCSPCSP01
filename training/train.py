"""
UAFN Training Script
=====================
Trains GCN, GraphSAGE, and MLP baseline models on air quality data.

Usage:
    python training/train.py [--epochs 40] [--lr 0.001] [--window 6]
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_generator import generate_full_dataset
from models.gcn_model import GCN_AQ
from models.sage_model import SAGE_AQ
from models.mlp_baseline import MLP_Baseline
from training.graph_builder import build_knn_graph, create_sequences
from training.evaluate import evaluate_model


def train_model(model, X_train, Y_train, X_val, Y_val,
                edge_index, edge_weight, epochs=40, lr=0.001,
                batch_size=64, model_name="Model"):
    """Train a model and return loss histories."""
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        indices = torch.randperm(len(X_train))
        for start in range(0, len(X_train), batch_size):
            idx = indices[start:start + batch_size]
            xb = X_train[idx]
            yb = Y_train[idx]
            
            pred = model(xb, edge_index, edge_weight)
            loss = F.mse_loss(pred, yb)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_train = epoch_loss / n_batches
        train_losses.append(avg_train)
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val, edge_index, edge_weight)
            val_loss = F.mse_loss(val_pred, Y_val).item()
            val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  {model_name} Epoch {epoch+1:3d}: "
                  f"train={avg_train:.4f}, val={val_loss:.4f}")
    
    return train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description="Train UAFN models")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--stations", type=int, default=10)
    parser.add_argument("--hours", type=int, default=360)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Generate data
    print("\n--- Generating synthetic data ---")
    aq_data, station_coords, meteo = generate_full_dataset(
        n_stations=args.stations, n_hours=args.hours
    )
    
    # 2. Build graph
    print("--- Building spatial graph ---")
    edge_index, edge_weight, G = build_knn_graph(station_coords, k=args.k)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 3. Preprocess PM2.5
    pm25 = pd.DataFrame(aq_data['PM2.5']).ffill().bfill().values
    scaler = StandardScaler()
    pm25_norm = scaler.fit_transform(pm25)
    
    X, Y = create_sequences(pm25_norm, args.window, horizon=1)
    
    split_train = int(0.70 * len(X))
    split_val = int(0.85 * len(X))
    
    X_train = torch.tensor(X[:split_train], dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y[:split_train], dtype=torch.float32).to(device)
    X_val = torch.tensor(X[split_train:split_val], dtype=torch.float32).to(device)
    Y_val = torch.tensor(Y[split_train:split_val], dtype=torch.float32).to(device)
    X_test = torch.tensor(X[split_val:], dtype=torch.float32).to(device)
    Y_test = torch.tensor(Y[split_val:], dtype=torch.float32).to(device)
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # 4. Train models
    models = {
        "GCN": GCN_AQ(1, 32, 16).to(device),
        "GraphSAGE": SAGE_AQ(1, 32, 16).to(device),
        "MLP": MLP_Baseline(args.window, args.stations, 64).to(device),
    }
    
    all_results = {}
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        train_model(model, X_train, Y_train, X_val, Y_val,
                   edge_index, edge_weight, epochs=args.epochs,
                   lr=args.lr, model_name=name)
        
        metrics = evaluate_model(model, X_test, Y_test, edge_index,
                                edge_weight, scaler)
        all_results[name] = metrics
        print(f"  Test: RMSE={metrics['RMSE']:.3f}, "
              f"MAE={metrics['MAE']:.3f}, R²={metrics['R2']:.3f}")
    
    # 5. Summary
    print("\n" + "=" * 55)
    print("FINAL RESULTS — PM2.5 1-hour ahead forecasting")
    print("=" * 55)
    df = pd.DataFrame(all_results).T
    print(df.round(4).to_string())
    print("=" * 55)


if __name__ == "__main__":
    main()
