# TRM - Architecture Mistral 7B ↔ BERT

**Version** : POC  
**Date** : Décembre 2025

---

## 📁 Structure du Projet

```
TRM/
├── Backend/
│   ├── Notebooks/      # Notebooks Jupyter pour expérimentation
│   └── Prod/           # Code de production
├── Frontend/           # Interface utilisateur
└── docs/              # Documentation
    ├── guides/         # Guides d'utilisation
    ├── references/    # Documentation de référence
    ├── logs/          # Logs de développement
    ├── notes/         # Notes de développement
    ├── supports/      # Documentation de support
    └── tutos/         # Tutoriels
```

---

## 🎯 Objectif

Architecture collaborative permettant à un modèle génératif (Mistral 7B) de traiter efficacement un contexte court (~500 tokens) en s'appuyant sur une mémoire condensée et pondérée fournie par un modèle encodeur (BERT), avec intégration RAG par philosophe.

---

## 📚 Documentation

- **PRD Enrichi** : `docs/guides/PRD_TRM_ENRICHI.md` (à créer ou copier depuis bergsonAndFriends)
- **Architecture** : Voir PRD pour détails complets

---

## 🚀 Quick Start

Voir la roadmap dans `docs/guides/PRD_TRM_ENRICHI.md` pour l'implémentation du POC.

