#!/usr/bin/env python3
"""
Push FastAI multilabel beetle classification model to Hugging Face Hub.

Covers the joint Cicindela + Platydracus taxonomic classifier. Uploads both the
FastAI learner and PyTorch model weights.

Usage:
    python push_to_hf.py [--model-path MODEL_PATH] [--repo-id REPO_ID]

Author: B. de Medeiros, 2025
"""

import os
import json
import copy
import argparse
from pathlib import Path
import torch
from fastai.vision.all import load_learner
from huggingface_hub import HfApi, push_to_hub_fastai
from timm.models.hub import push_to_hf_hub
from timm import create_model

def parse_args():
    parser = argparse.ArgumentParser(description='Push FastAI beetle classification model to Hugging Face Hub')
    parser.add_argument('--model-path', type=str,
                       default='exported_fastai_models/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k_ml.pkl',
                       help='Path to the FastAI model pkl file')
    parser.add_argument('--model-card', type=str,
                       default='hf_model_card.md',
                       help='Path to the model card markdown file')
    parser.add_argument('--repo-id', type=str,
                       default='brunoasm/Cicindela_Platydracus_ID_FMNH',
                       help='HuggingFace repository ID')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print configuration without uploading')
    return parser.parse_args()


def classify_label(label):
    """Return 'genus' / 'species_group' / 'species' / 'subspecies' for a vocabulary entry."""
    if not label:
        return 'species'
    if label[:1].isupper():
        return 'genus'
    if 'group_' in label:
        return 'species_group'
    return 'species' if label.count('_') == 1 else 'subspecies'

def extract_vocab_from_learner(learn):
    """Extract the full union vocabulary from the FastAI learner.

    Multi-head models store labels inside a MultiHeadTarget transform rather
    than directly on the dataloaders, so we recover them from there.
    """
    # Multi-head path: find the MultiHeadTarget transform buried in dls.tls
    mht = None
    for tl in learn.dls.tls:
        for t in tl.tfms.fs:
            if type(t).__name__ == 'MultiHeadTarget':
                mht = t
                break
        if mht is not None:
            break
    if mht is not None:
        vocab_g = list(mht.vocab_g)
        vocab_cic = [l for l, _ in sorted(mht.cic2idx.items(), key=lambda kv: kv[1])]
        vocab_platy = [l for l, _ in sorted(mht.platy2idx.items(), key=lambda kv: kv[1])]
        return vocab_g + vocab_cic + vocab_platy

    # Single-head fallback: try standard fastai vocab attributes
    for attr in ('vocab', 'vocab_g'):
        try:
            raw = getattr(learn.dls, attr)
            try:
                return list(raw.items())
            except (TypeError, AttributeError):
                return [str(v) for v in raw]
        except AttributeError:
            continue

    print("Warning: Could not extract vocabulary from learn.dls")
    return None

def create_taxon_config(learn, base_config, repo_id):
    """
    Create configuration for the multi-genus beetle classifier (Cicindela +
    Platydracus) based on the base timm config and the trained FastAI learner.
    """
    vocab = extract_vocab_from_learner(learn)
    num_classes = len(vocab) if vocab is not None else None

    if vocab:
        genus_count = sum(1 for l in vocab if classify_label(l) == 'genus')
        species_group_count = sum(1 for l in vocab if classify_label(l) == 'species_group')
        species_count = sum(1 for l in vocab if classify_label(l) == 'species')
        subspecies_count = sum(1 for l in vocab if classify_label(l) == 'subspecies')
        genera = sorted(l for l in vocab if classify_label(l) == 'genus')
    else:
        genus_count = species_group_count = species_count = subspecies_count = None
        genera = None

    config = {
        "architecture": "eva02_large_patch14_448",
        "num_classes": num_classes,
        "num_features": 1024,
        "global_pool": "avg",
        "pretrained_cfg": {
            "hf_hub_id": repo_id,
            "source": "hf-hub",
            "architecture": "hf-hub:timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
            "tag": "Cicindela_Platydracus_multilabel_classification",
            "custom_load": False,
            "input_size": [3, 448, 448],
            "fixed_input_size": True,
            "interpolation": "bicubic",
            "crop_pct": 1.0,
            "crop_mode": "squash",
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
            "num_classes": num_classes,
            "pool_size": None,
            "first_conv": "patch_embed.proj",
            "classifier": "head",
            "license": "mit"
        },
        "model_type": "multilabel_classification",
        "task": "taxonomic_classification",
        "dataset": "FMNH_Cicindela_Platydracus_specimens",
        "genera": genera,
        "genus_count": genus_count,
        "species_group_count": species_group_count,
        "species_count": species_count,
        "subspecies_count": subspecies_count,
        "label_threshold": 0.5,
        "labels": vocab if vocab else None,
        "training_framework": "fastai",
        "base_model": "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
    }

    return config

def save_pytorch_model(learn, model_path):
    """Extract and save PyTorch model weights from FastAI learner"""
    # Get the underlying PyTorch model
    pytorch_model = learn.model
    
    # Save the state dict
    pytorch_path = model_path.parent / "pytorch_model.bin"
    torch.save(pytorch_model.state_dict(), pytorch_path)
    
    return pytorch_path

def main():
    args = parse_args()
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading FastAI learner from: {model_path}")
    
    # Load the FastAI learner
    learn = load_learner(model_path, cpu=False)
    
    # Reset metrics and recorder to clean up the model for export
    for m in learn.metrics:
        if hasattr(m, 'reset'):
            m.reset()
    
    if hasattr(learn.recorder, 'reset'):
        learn.recorder.reset()
    
    # Extract vocabulary and create configuration
    vocab = extract_vocab_from_learner(learn)
    print(f"Model vocabulary size: {len(vocab) if vocab else 'Unknown'}")
    if vocab:
        print(f"Genus labels:         {sum(1 for l in vocab if classify_label(l) == 'genus')}")
        print(f"Species group labels: {sum(1 for l in vocab if classify_label(l) == 'species_group')}")
        print(f"Species labels:       {sum(1 for l in vocab if classify_label(l) == 'species')}")
        print(f"Subspecies labels:    {sum(1 for l in vocab if classify_label(l) == 'subspecies')}")
    
    # Base configuration from the pretrained timm model
    base_config = {
        "architecture": "eva02_large_patch14_448",
        "num_classes": 1000,
        "num_features": 1024,
        "global_pool": "avg",
        "pretrained_cfg": {
            "tag": "mim_m38m_ft_in22k_in1k",
            "custom_load": False,
            "input_size": [3, 448, 448],
            "fixed_input_size": True,
            "interpolation": "bicubic",
            "crop_pct": 1.0,
            "crop_mode": "squash",
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
            "num_classes": 1000,
            "pool_size": None,
            "first_conv": "patch_embed.proj",
            "classifier": "head",
            "license": "mit"
        }
    }
    
    # Create custom configuration
    config = create_taxon_config(learn, base_config, args.repo_id)
    
    model_card_path = Path(args.model_card)
    if not model_card_path.exists():
        print(f"Warning: model card not found at {model_card_path}, skipping README upload")
        model_card_path = None

    if args.dry_run:
        print("\nDry run - Configuration that would be uploaded:")
        print(json.dumps(config, indent=2))
        print(f"\nWould upload to repository: {args.repo_id}")
        if model_card_path:
            print(f"Would upload model card from: {model_card_path}")
        return
    
    print(f"Uploading to repository: {args.repo_id}")
    
    # Create temporary directory for additional files
    temp_dir = model_path.parent / "temp_hf_upload"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Save configuration
        config_path = temp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save PyTorch model weights
        print("Extracting PyTorch model weights...")
        pytorch_path = save_pytorch_model(learn, temp_dir)
        
        # Save vocabulary as separate file
        if vocab:
            vocab_path = temp_dir / "vocab.json"
            with open(vocab_path, 'w') as f:
                json.dump(vocab, f, indent=2)
        
        # Upload FastAI model
        print("Uploading FastAI learner...")
        push_to_hub_fastai(
            learner=learn,
            repo_id=args.repo_id,
            commit_message="Upload Cicindela+Platydracus multilabel classification model (FastAI, 4-level hierarchy)"
        )
        
        # Upload additional files using HfApi
        print("Uploading configuration and PyTorch weights...")
        api = HfApi()
        
        # Upload config
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="config.json",
            repo_id=args.repo_id,
            commit_message="Add model configuration"
        )
        
        # Upload PyTorch weights
        api.upload_file(
            path_or_fileobj=str(pytorch_path),
            path_in_repo="pytorch_model.bin", 
            repo_id=args.repo_id,
            commit_message="Add PyTorch model weights"
        )
        
        # Upload vocabulary if available
        if vocab:
            api.upload_file(
                path_or_fileobj=str(vocab_path),
                path_in_repo="vocab.json",
                repo_id=args.repo_id,
                commit_message="Add model vocabulary"
            )

        # Upload model card
        if model_card_path:
            print("Uploading model card...")
            api.upload_file(
                path_or_fileobj=str(model_card_path),
                path_in_repo="README.md",
                repo_id=args.repo_id,
                commit_message="Update model card for Cicindela+Platydracus joint classifier"
            )

        print(f"Successfully uploaded model to: https://huggingface.co/{args.repo_id}")
        
    finally:
        # Cleanup temporary files
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    print("Upload complete!")

if __name__ == "__main__":
    main()
