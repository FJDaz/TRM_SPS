# ⚡ TRM POC - Démarrage Rapide (5 min)

**Vous voulez démarrer immédiatement ?** Suivez ce guide !

---

## 🎯 Objectif

Valider l'architecture TRM (BERT + Mistral 7B) en 3 notebooks Colab gratuits, **sans dépenser 1€**.

---

## 📝 3 Étapes - 1 Journée

### 1️⃣ BERT Encoder (1-2h) - **COMMENCEZ ICI**

```bash
👉 Ouvrez: notebooks/01_BERT_Encoder_STATE_IMAGE.ipynb dans Colab
```

**Actions:**
1. Cliquez sur "Ouvrir dans Colab" (ou upload le fichier)
2. Runtime > Run all
3. Attendez ~5 min (installation + tests)
4. Vérifiez que tous les tests sont ✅
5. Téléchargez `state_image_example.json`

**Résultat:** BERT Encoder fonctionnel + STATE_IMAGE validé

---

### 2️⃣ RAG Embeddings (1-2h)

```bash
👉 Ouvrez: notebooks/02_RAG_Embeddings_Generation.ipynb dans Colab
```

**Actions:**
1. **IMPORTANT:** Runtime > Change runtime type > **T4 GPU**
2. Upload les 6 fichiers corpus (ou clonez le repo GitHub)
3. Runtime > Run all
4. Attendez ~30-60 min (génération embeddings)
5. **CRITIQUE:** Téléchargez `rag_exports.zip` (nécessaire pour Phase 1)

**Résultat:** Index RAG FAISS pour 3 philosophes

---

### 3️⃣ Mistral 7B Testing (2-3h)

```bash
👉 Ouvrez: notebooks/03_Mistral_7B_Testing.ipynb dans Colab
```

**Actions:**
1. **IMPORTANT:** Runtime > Change runtime type > **T4 GPU**
2. Runtime > Run all
3. ⏳ Attendez chargement Mistral 7B (~5-10 min)
4. Observez benchmarks 5 scénarios
5. Téléchargez `mistral_benchmark_results.json`

**Résultat:** Mistral 7B fonctionnel + validation ≤500 tokens

---

## ✅ Validation Rapide

Après les 3 notebooks, vérifiez:

- [ ] BERT génère STATE_IMAGE avec 9 axes ✅
- [ ] STATE_IMAGE ≤250 tokens ✅
- [ ] RAG retrieval retourne passages pertinents ✅
- [ ] Mistral contexte ≤500 tokens (tous scénarios) ✅
- [ ] Latence Mistral <3s ✅

**Si tous ✅ → Phase 0 validée !**

---

## 💰 Prochaine Étape: Phase 1 (100€)

Une fois Phase 0 terminée:

1. **Upload sur Vast.ai:**
   - `rag_exports.zip` (index RAG)
   - Code BERT + Mistral

2. **Setup 2 instances:**
   - CPU pour BERT (~$0.05/h)
   - GPU RTX 4090 pour Mistral 7B (~$0.30/h)

3. **Intégration complète:**
   - Pipeline BERT → RAG → Mistral
   - Tests end-to-end
   - Benchmarks comparatifs vs Qwen 14B

**Budget:** ~80-100€ pour POC complet (7 jours)

---

## 🆘 Besoin d'Aide ?

### Colab déconnecte ?
→ Voir `README_PHASE_0.md` section Troubleshooting

### GPU T4 pas disponible ?
→ Attendre 5-10 min ou essayer Kaggle (GPU gratuit)

### Upload corpus échoue ?
→ Cloner repo GitHub directement dans Colab

---

## 📚 Documentation Complète

- [README Phase 0](README_PHASE_0.md) - Instructions détaillées
- [PRD TRM](docs/guides/PRD_TRM_ENRICHI.md) - Architecture complète

---

**Temps total Phase 0:** 1 journée (4-7h)
**Coût:** 0€
**Résultat:** POC TRM validé, prêt pour infrastructure payante

🚀 **Lancez-vous maintenant !**
