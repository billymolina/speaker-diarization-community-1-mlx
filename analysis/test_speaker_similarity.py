"""
Test if speaker similarity relationships are preserved between PyTorch and MLX models.

This is the most important test - even if embeddings differ slightly in absolute values,
what matters for speaker diarization is that similarity relationships are preserved:
- Similar speakers should have high cosine similarity in both models
- Different speakers should have low cosine similarity in both models
"""

import numpy as np
import mlx.core as mx
import torch


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    print("="*70)
    print(" Speaker Similarity Preservation Test")
    print("="*70)

    # Load models
    from pyannote.audio import Model
    pt_model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    pt_resnet = pt_model.resnet
    pt_resnet.eval()

    from src.resnet_embedding import load_resnet34_embedding
    mlx_model = load_resnet34_embedding("models/embedding_mlx/weights.npz")

    # Create 4 different "speakers" (random mel spectrograms with different seeds)
    print("\n[INFO] Generating embeddings for 4 simulated speakers...")

    speakers_pt = []
    speakers_mlx = []

    for speaker_id in range(4):
        np.random.seed(speaker_id * 100)
        mel_np = np.random.randn(1, 150, 80).astype(np.float32)

        # PyTorch
        mel_pt = torch.from_numpy(mel_np)
        with torch.no_grad():
            pt_output = pt_resnet(mel_pt)
            if isinstance(pt_output, tuple):
                pt_emb = pt_output[1]
            else:
                pt_emb = pt_output
            speakers_pt.append(pt_emb.detach().cpu().numpy()[0])

        # MLX
        mel_mlx = mx.array(mel_np)
        mlx_emb = mlx_model(mel_mlx)
        speakers_mlx.append(np.array(mlx_emb)[0])

        print(f"  Speaker {speaker_id}: PT norm={np.linalg.norm(speakers_pt[-1]):.4f}, "
              f"MLX norm={np.linalg.norm(speakers_mlx[-1]):.4f}")

    # Compute all pairwise similarities
    print("\n" + "="*70)
    print(" Pairwise Cosine Similarities")
    print("="*70)

    print("\nPyTorch Model:")
    print("     ", end="")
    for j in range(4):
        print(f"  Spk{j}  ", end="")
    print()

    pt_sim_matrix = np.zeros((4, 4))
    for i in range(4):
        print(f"Spk{i}:", end="")
        for j in range(4):
            sim = cosine_similarity(speakers_pt[i], speakers_pt[j])
            pt_sim_matrix[i, j] = sim
            print(f"  {sim:6.4f}", end="")
        print()

    print("\nMLX Model:")
    print("     ", end="")
    for j in range(4):
        print(f"  Spk{j}  ", end="")
    print()

    mlx_sim_matrix = np.zeros((4, 4))
    for i in range(4):
        print(f"Spk{i}:", end="")
        for j in range(4):
            sim = cosine_similarity(speakers_mlx[i], speakers_mlx[j])
            mlx_sim_matrix[i, j] = sim
            print(f"  {sim:6.4f}", end="")
        print()

    # Compute difference in similarity matrices
    sim_diff = np.abs(pt_sim_matrix - mlx_sim_matrix)
    max_sim_diff = np.max(sim_diff)
    mean_sim_diff = np.mean(sim_diff)

    print("\n" + "="*70)
    print(" Similarity Matrix Comparison")
    print("="*70)
    print(f"Max abs difference: {max_sim_diff:.6f}")
    print(f"Mean abs difference: {mean_sim_diff:.6f}")

    print("\nDifference Matrix:")
    print("     ", end="")
    for j in range(4):
        print(f"  Spk{j}  ", end="")
    print()
    for i in range(4):
        print(f"Spk{i}:", end="")
        for j in range(4):
            print(f"  {sim_diff[i, j]:6.4f}", end="")
        print()

    # Test if similarity ranking is preserved
    print("\n" + "="*70)
    print(" Similarity Ranking Preservation")
    print("="*70)

    ranking_preserved = 0
    total_comparisons = 0

    for speaker in range(4):
        # Get similarities to other speakers
        pt_sims = [(j, pt_sim_matrix[speaker, j]) for j in range(4) if j != speaker]
        mlx_sims = [(j, mlx_sim_matrix[speaker, j]) for j in range(4) if j != speaker]

        # Sort by similarity (descending)
        pt_ranking = [x[0] for x in sorted(pt_sims, key=lambda x: x[1], reverse=True)]
        mlx_ranking = [x[0] for x in sorted(mlx_sims, key=lambda x: x[1], reverse=True)]

        print(f"\nSpeaker {speaker} most similar to:")
        print(f"  PyTorch: {pt_ranking}")
        print(f"  MLX:     {mlx_ranking}")

        # Check if rankings match
        if pt_ranking == mlx_ranking:
            ranking_preserved += 1
            print(f"  ✓ Rankings match!")
        else:
            print(f"  ✗ Rankings differ")

        total_comparisons += 1

    # Final verdict
    print("\n" + "="*70)
    print(" FINAL VERDICT")
    print("="*70)

    print(f"\nSimilarity matrix differences:")
    print(f"  Max abs diff: {max_sim_diff:.6f}")
    print(f"  Mean abs diff: {mean_sim_diff:.6f}")

    print(f"\nRanking preservation:")
    print(f"  {ranking_preserved}/{total_comparisons} speakers have matching similarity rankings")

    if max_sim_diff < 0.05 and ranking_preserved >= 0.75 * total_comparisons:
        print(f"\n✓ PASS: Speaker similarity is well preserved!")
        print(f"   The MLX model should work well for speaker diarization.")
        return 0
    elif max_sim_diff < 0.15:
        print(f"\n⚠ ACCEPTABLE: Speaker similarity is reasonably preserved")
        print(f"   The MLX model may work for speaker diarization but with some degradation.")
        return 0
    else:
        print(f"\n✗ FAIL: Speaker similarity is not well preserved")
        print(f"   The MLX model may not work reliably for speaker diarization.")
        return 1


if __name__ == "__main__":
    exit(main())
