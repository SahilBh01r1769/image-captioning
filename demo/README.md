# Optional checkpoint diagnostic

This Streamlit view is deliberately outside CaptionLab's experimental evidence path. It can load a local custom checkpoint and vocabulary, generate greedy or beam-search captions for an uploaded image, show selected-token probabilities, and overlay spatial attention weights.

It does not:

- provide a hosted third-party captioner;
- contribute metrics or examples to the controlled comparison;
- turn attention weights into causal explanations;
- represent a production or deployment claim.

Place `best_model.pth` and `vocabulary.pkl` under `models/`, then run:

```bash
streamlit run streamlit_app.py
```

Use the full test evaluator for project results. This viewer is only useful for manually inspecting a checkpoint after the quantitative protocol is complete.
