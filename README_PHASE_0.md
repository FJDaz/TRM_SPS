# 🚀 TRM POC - Phase 0 (Colab Gratuit - 0€)

**Objectif:** Développer et tester tous les composants TRM sans infrastructure payante

**Budget:** 0€ (Google Colab gratuit)

**Durée:** 1-2 jours développement

---

## 📋 Checklist Phase 0

### ✅ Composants à Développer

- [ ] **BERT Encoder** (Notebook 1) - CPU Colab
  - Extraction concepts conversationnels
  - Extraction concepts RAG condensés
  - Analyse multi-axes (intention, tension, style, etc.)
  - Génération STATE_IMAGE JSON structuré

- [ ] **RAG Embeddings** (Notebook 2) - GPU T4 Colab
  - Génération embeddings 3 philosophes (Spinoza/Bergson/Kant)
  - Indexation FAISS
  - Export index pour réutilisation

- [ ] **Mistral 7B Generator** (Notebook 3) - GPU T4 Colab
  - Chargement Mistral 7B (4-bit quantization)
  - Génération avec STATE_IMAGE
  - Validation contrainte ≤500 tokens
  - Benchmarks latence

---

## 🗂️ Structure Notebooks

```
TRM/
├── notebooks/
│   ├── 01_BERT_Encoder_STATE_IMAGE.ipynb       (CPU Colab)
│   ├── 02_RAG_Embeddings_Generation.ipynb      (GPU T4 Colab)
│   └── 03_Mistral_7B_Testing.ipynb             (GPU T4 Colab)
├── data/
│   └── (fichiers corpus à uploader)
└── README_PHASE_0.md (ce fichier)
```

---

## 📝 Instructions Étape par Étape

### Étape 1: BERT Encoder (1-2h)

**Fichier:** `notebooks/01_BERT_Encoder_STATE_IMAGE.ipynb`

**Runtime:** CPU (gratuit, illimité)

**Actions:**
1. Ouvrir dans Google Colab
2. Exécuter toutes les cellules (Runtime > Run all)
3. Vérifier les tests unitaires passent ✅
4. Télécharger `state_image_example.json`

**Livrables:**
- ✅ Classe `BERTEncoder` fonctionnelle
- ✅ STATE_IMAGE structuré validé
- ✅ Taille ≤250 tokens

---

### Étape 2: RAG Embeddings (1-2h)

**Fichier:** `notebooks/02_RAG_Embeddings_Generation.ipynb`

**Runtime:** GPU T4 (gratuit, sessions 12h max)

**Actions:**
1. Ouvrir dans Google Colab
2. Activer GPU: Runtime > Change runtime type > T4 GPU
3. **IMPORTANT:** Uploader les 6 fichiers corpus depuis `bergsonAndFriends/data/RAG/`:
   ```
   - Corpus Spinoza Dialogique 18k - Éthique II-IV.md
   - Glossaire Conversationnel Spinoza - 12 Concepts.md
   - corpus_bergson_27k_dialogique.md
   - glossaire_bergson_conversationnel.md
   - corpus_kant_20k.txt.md
   - glossaire_kant_conversationnel.md
   ```
4. Exécuter toutes les cellules
5. Télécharger `rag_exports.zip` (contient index FAISS + passages)

**Livrables:**
- ✅ Embeddings 3 philosophes générés
- ✅ Index FAISS opérationnels
- ✅ Retrieval sémantique fonctionnel

**⚠️ Note:** Sauvegarder `rag_exports.zip` localement - nécessaire pour Phase 1 (Vast.ai)

---

### Étape 3: Mistral 7B Testing (2-3h)

**Fichier:** `notebooks/03_Mistral_7B_Testing.ipynb`

**Runtime:** GPU T4 (gratuit, sessions 12h max)

**Actions:**
1. Ouvrir dans Google Colab
2. Activer GPU: Runtime > Change runtime type > T4 GPU
3. Exécuter toutes les cellules
4. ⏳ Attendre chargement Mistral 7B (5-10 min)
5. Vérifier benchmarks 5 scénarios
6. Télécharger `mistral_benchmark_results.json`

**Livrables:**
- ✅ Mistral 7B fonctionnel (4-bit)
- ✅ Génération avec STATE_IMAGE validée
- ✅ Contrainte ≤500 tokens respectée
- ✅ Benchmarks latence (<3s)

---

## 📊 Validation Phase 0

### Métriques Cibles

| Métrique | Objectif | Critique |
|----------|----------|----------|
| **STATE_IMAGE taille** | 150-250 tokens | ≤250 |
| **Contexte Mistral** | ≤500 tokens | **✅ Critique POC** |
| **Latence génération** | <3s | <5s |
| **RAG retrieval** | 2-3 passages | Top-3 pertinents |
| **Concepts extraits** | 3-8 par tour | >0 |

### Tests de Validation

**BERT Encoder:**
- [ ] Extraction keywords fonctionne
- [ ] Concepts RAG condensés (pas texte brut)
- [ ] STATE_IMAGE JSON valide (9 axes)
- [ ] Taille ≤250 tokens

**RAG Embeddings:**
- [ ] Index FAISS créés pour 3 philosophes
- [ ] Retrieval retourne passages pertinents
- [ ] Similarity scores ≥0.45

**Mistral 7B:**
- [ ] Modèle chargé en <10min
- [ ] VRAM utilisée <15GB (T4)
- [ ] Contexte ≤500 tokens (tous scénarios)
- [ ] Génération cohérente avec STATE_IMAGE

---

## 🎯 Objectifs Phase 0

### ✅ Réussite Phase 0 = 3 Livrables

1. **BERT Encoder** fonctionnel + STATE_IMAGE validé
2. **RAG Embeddings** générés + index FAISS exportés
3. **Mistral 7B** testé + benchmarks latence validés

### ➡️ Phase 1 (Vast.ai - 100€)

Une fois Phase 0 validée:
1. Upload `rag_exports.zip` sur Vast.ai
2. Setup 2 instances (CPU BERT + GPU Mistral)
3. Intégration pipeline complet
4. Benchmarks comparatifs TRM vs Qwen 14B

---

## 💰 Économies Phase 0

| Tâche | Coût Vast.ai | Coût Colab | Économie |
|-------|--------------|------------|----------|
| Dev BERT Encoder | ~$5 (10h CPU) | 0€ (illimité) | **$5** |
| Gen RAG Embeddings | ~$10 (10h GPU) | 0€ (12h T4) | **$10** |
| Tests Mistral 7B | ~$15 (20h GPU) | 0€ (12h T4) | **$15** |
| **Total** | **~$30** | **0€** | **$30** |

**Impact:** Budget 100€ → 130€ effectif grâce à Phase 0

---

## ⚠️ Limites Colab Gratuit

### Ce qui fonctionne ✅
- Dev BERT Encoder (CPU illimité)
- Gen RAG embeddings (T4 12h suffit)
- Tests Mistral 7B isolé (T4 12h suffit)

### Ce qui ne fonctionne PAS ❌
- **Intégration BERT + Mistral simultanée** (besoin 2 instances séparées)
- **Tests end-to-end pipeline complet** (nécessite Vast.ai)
- **Benchmarks comparatifs avec Qwen 14B** (trop lourd pour T4)
- **Persistance modèles chargés** (sessions limitées 12h)

**→ Phase 1 (Vast.ai) obligatoire pour POC complet**

---

## 📚 Ressources

### Documentation
- [Architecture TRM (PRD)](docs/guides/PRD_TRM_ENRICHI.md)
- [Corpus philosophiques](../bergsonAndFriends/data/RAG/)

### Liens Utiles
- [Google Colab](https://colab.research.google.com/)
- [Sentence-Transformers](https://www.sbert.net/)
- [Mistral AI](https://mistral.ai/)

---

## 🆘 Troubleshooting

### Problème: Colab déconnecte avant la fin
**Solution:** Ouvrir console navigateur et exécuter:
```javascript
function ClickConnect(){
  console.log("Keep alive");
  document.querySelector("colab-toolbar-button").click()
}
setInterval(ClickConnect, 60000)
```

### Problème: T4 pas disponible
**Solution:**
1. Attendre 5-10 min
2. Ou passer à Colab Pro ($10/mois) pour GPU prioritaire
3. Ou utiliser Kaggle (15h GPU gratuit/semaine)

### Problème: Upload corpus échoue
**Solution:**
1. Cloner repo GitHub directement dans Colab:
```python
!git clone https://github.com/YOUR_USERNAME/bergsonAndFriends.git
CORPUS_DIR = "bergsonAndFriends/data/RAG/"
```

### Problème: Mistral 7B OOM (Out of Memory)
**Solution:**
1. Vérifier quantization 4-bit activée
2. Réduire `max_new_tokens` à 200
3. Redémarrer runtime (Runtime > Restart runtime)

---

## ✅ Checklist Finale Phase 0

Avant de passer à Phase 1 (Vast.ai), vérifier:

- [ ] Les 3 notebooks exécutés sans erreur
- [ ] `state_image_example.json` téléchargé
- [ ] `rag_exports.zip` téléchargé (critique pour Phase 1)
- [ ] `mistral_benchmark_results.json` téléchargé
- [ ] Tous les tests de validation passés
- [ ] Contexte Mistral ≤500 tokens validé

**Si tous ✅ → Prêt pour Phase 1 (100€)**

---

**Dernière mise à jour:** Décembre 2025
**Version:** 1.0
