#!/usr/bin/env python3
"""
Debug script to diagnose why interaction training starts with impossibly low loss.

Add this code to train_interaction.py right after creating the model and before training.
"""

import torch
import torch.nn.functional as F

def debug_interaction_model(model, train_loader, device):
    """
    Check if the model is actually initialized randomly or if something is wrong.
    """
    print("\n" + "="*80)
    print("DEBUGGING INTERACTION MODEL - EPOCH 0 (Before Training)")
    print("="*80)
    
    # Get one batch
    batch = next(iter(train_loader))
    inputs, targets = batch
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    targets = {k: v.to(device) for k, v in targets.items()}
    
    # Get model predictions (random initialization)
    model.eval()
    with torch.no_grad():
        encoder_kw = {
            k: inputs[k] for k in ["stem_ids", "branch_ids", "stem_elements", "stem_polarities", "pillar_positions"]
            if k in inputs
        }
        natal_emb = model.encoder(**encoder_kw)
        logits = model.head(natal_emb)
        
        target = targets["interaction_vector"]
        
        print(f"\n1. BATCH INFO:")
        print(f"   Batch size: {target.shape[0]}")
        print(f"   Num bits: {target.shape[1]}")
        
        print(f"\n2. TARGET (GROUND TRUTH) STATS:")
        print(f"   Shape: {target.shape}")
        print(f"   Dtype: {target.dtype}")
        print(f"   Unique values: {target.unique().tolist()}")
        print(f"   Mean (positive rate): {target.float().mean().item():.4f}")
        print(f"   First sample: {target[0][:10].tolist()} ... (first 10 bits)")
        
        print(f"\n3. LOGITS (MODEL OUTPUT) STATS:")
        print(f"   Shape: {logits.shape}")
        print(f"   Dtype: {logits.dtype}")
        print(f"   Range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        print(f"   Mean: {logits.mean().item():.4f}")
        print(f"   Std: {logits.std().item():.4f}")
        print(f"   First sample: {logits[0][:10].tolist()}")
        
        print(f"\n4. PROBABILITIES (after sigmoid):")
        probs = torch.sigmoid(logits)
        print(f"   Mean: {probs.mean().item():.4f} (should be ~0.5 for random)")
        print(f"   Std: {probs.std().item():.4f} (should be ~0.1-0.2 for random)")
        print(f"   Range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
        print(f"   First sample: {probs[0][:10].tolist()}")
        
        print(f"\n5. LOSS CALCULATION:")
        
        # Check if pos_weight exists and what it is
        pos_weight = getattr(model.head, "_pos_weight", None)
        if pos_weight is not None:
            print(f"   pos_weight exists: shape={pos_weight.shape}")
            print(f"   pos_weight range: [{pos_weight.min().item():.4f}, {pos_weight.max().item():.4f}]")
            print(f"   pos_weight mean: {pos_weight.mean().item():.4f}")
            print(f"   pos_weight first 10: {pos_weight[:10].tolist()}")
        else:
            print(f"   pos_weight: None (no weighting)")
        
        # Manual BCE calculation (no pos_weight)
        bce_no_weight = F.binary_cross_entropy_with_logits(
            logits, target, reduction='none'
        )
        print(f"\n   BCE (no pos_weight):")
        print(f"     Per-bit mean: {bce_no_weight.mean().item():.6f}")
        print(f"     Expected for random: ~0.693")
        print(f"     Total (sum): {bce_no_weight.sum().item():.6f}")
        print(f"     Per-sample mean: {bce_no_weight.mean(dim=1).mean().item():.6f}")
        
        # BCE with pos_weight (if exists)
        if pos_weight is not None:
            bce_with_weight = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pos_weight, reduction='none'
            )
            print(f"\n   BCE (WITH pos_weight):")
            print(f"     Per-bit mean: {bce_with_weight.mean().item():.6f}")
            print(f"     Total (sum): {bce_with_weight.sum().item():.6f}")
            print(f"     Ratio to no-weight: {bce_with_weight.mean().item() / bce_no_weight.mean().item():.2f}x")
        
        # Use the model's actual loss function
        loss_from_head = model.head.loss(logits, target, reduction='mean')
        print(f"\n   Loss from head.loss():")
        print(f"     Value: {loss_from_head.item():.6f}")
        
        # Check encoder embedding statistics
        print(f"\n6. ENCODER EMBEDDING STATS:")
        print(f"   Shape: {natal_emb.shape}")
        print(f"   Mean: {natal_emb.mean().item():.4f}")
        print(f"   Std: {natal_emb.std().item():.4f}")
        print(f"   Range: [{natal_emb.min().item():.4f}, {natal_emb.max().item():.4f}]")
        
        # Check if embeddings are collapsed (all the same)
        if natal_emb.std().item() < 0.01:
            print(f"   ⚠️  WARNING: Encoder embeddings have very low variance!")
            print(f"   ⚠️  This suggests encoder is not initialized properly.")
        
        # Check head weights
        print(f"\n7. HEAD LINEAR LAYER WEIGHTS:")
        head_weight = model.head.linear.weight  # shape: (64, 256)
        print(f"   Weight shape: {head_weight.shape}")
        print(f"   Weight mean: {head_weight.mean().item():.6f}")
        print(f"   Weight std: {head_weight.std().item():.6f}")
        
        if model.head.linear.bias is not None:
            head_bias = model.head.linear.bias  # shape: (64,)
            print(f"   Bias shape: {head_bias.shape}")
            print(f"   Bias mean: {head_bias.mean().item():.6f}")
            print(f"   Bias std: {head_bias.std().item():.6f}")
            print(f"   Bias range: [{head_bias.min().item():.4f}, {head_bias.max().item():.4f}]")
            
            # Check if bias is causing the low loss
            if head_bias.abs().max().item() > 5.0:
                print(f"   ⚠️  WARNING: Head bias has large values (>{5.0})!")
                print(f"   ⚠️  This could bias predictions before training starts.")
    
    print("\n" + "="*80)
    print("END DEBUG - Check the values above")
    print("="*80 + "\n")
    
    # Interpretation
    print("INTERPRETATION:")
    if loss_from_head.item() < 0.5:
        print("❌ Loss is suspiciously low (<0.5)!")
        print("   Possible causes:")
        print("   1. pos_weight is creating extreme reweighting")
        print("   2. Head bias is pre-trained or initialized badly")
        print("   3. Encoder is producing non-random embeddings")
        print("   4. There's still data leakage somewhere")
    elif 0.5 <= loss_from_head.item() < 0.65:
        print("⚠️  Loss is slightly low (0.5-0.65)")
        print("   This might be okay, but check pos_weight effects")
    elif 0.65 <= loss_from_head.item() <= 0.75:
        print("✅ Loss looks normal (~0.693 for random)")
    else:
        print("⚠️  Loss is higher than expected (>0.75)")
        print("   Check if pos_weight is too aggressive")
    
    model.train()


# USAGE: Add this to train_interaction.py after line ~117 (after creating model, before trainer.run()):
# 
# debug_interaction_model(model, train_loader, device)
# import sys
# sys.exit(0)  # Exit after debug (remove this line to continue training)